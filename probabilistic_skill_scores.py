#!/usr/bin/env python3
"""
Probabilistic skill scores  for precipitation nowcasting ensembles. 
Pre processing for the computation of reliability diagrams, ROC curves, Brier core
and rank histograms.

Stages:
  1. Assemble obs and ensemble arrays per timestamp, crop to INCA domain (cached as .npz)
  2. Compute exceedance probability maps for the precipitation threshold
  3. Contingency tables (POD, FAR, freq-in-bin, reliability) + Brier score
  4. Rank histogram (Talagrand diagram)

Output: one NetCDF file with all metrics aggregated across timestamps.

Usage:
    python probabilistic_skill_scores.py \
        --run-dir /path/to/RUN5/NOWCAST.SIMULATIONS/ \
        --output-dir /path/to/output/ \
        --csv /path/to/filtered_timestamps.csv \
        [--threshold 0.1667 0.3333 0.8333 1.6667] \
        [--members 0 10] \
        [--prefix npc1.prob]
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import netCDF4 as nc
import rdata

# ============================================================
# FIXED CONSTANTS
# ============================================================
NR_FRAMES    = 36
N_LEAD       = NR_FRAMES + 1   # 37
DIM_X, DIM_Y = 710, 640

SWISS_CORNERS = (485,  75, 835, 295)
RADAR_CORNERS = (255, -160, 965, 480)
H             = 75

INCA_CORNERS = (
    SWISS_CORNERS[0] - H,
    SWISS_CORNERS[1] - H,
    SWISS_CORNERS[2] + H,
    SWISS_CORNERS[3] + H,
)

# R 1-indexed positions converted to Python 0-indexed slices
_x1_r = INCA_CORNERS[0] - RADAR_CORNERS[0]
_x2_r = _x1_r + (INCA_CORNERS[2] - INCA_CORNERS[0])
_y1_r = INCA_CORNERS[1] - RADAR_CORNERS[1]
_y2_r = _y1_r + (INCA_CORNERS[3] - INCA_CORNERS[1])

X_SL = slice(_x1_r - 1, _x2_r)
Y_SL = slice(_y1_r - 1, _y2_r)

INCA_NX  = _x2_r - _x1_r + 1   # 501
INCA_NY  = _y2_r - _y1_r + 1   # 371
N_PIXELS = INCA_NX * INCA_NY    # matches R's n = (x.INCA.2-x.INCA.1+1)*(y.INCA.2-y.INCA.1+1)

PROB_THRESHOLDS = np.arange(0.0, 1.0, 0.1)   # [0.0, 0.1, ..., 0.9]


# ============================================================
# FILE I/O
# ============================================================

def _load_rda(path):
    parsed = rdata.parser.parse_file(path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        return rdata.conversion.convert(parsed)


def _load_obs(path):
    """observations[[2]] — shape (710, 640, 3); we use [,,3] i.e. axis-2 index 2."""
    return np.array(_load_rda(path)['observations'][1])


def _load_future_obs(path):
    """observation.array.v — shape (710, 640, 36)."""
    return np.array(_load_rda(path)['observation.array.v'])


def _load_sim(path):
    """simulations[[2]] — shape (710, 640, 36)."""
    return np.array(_load_rda(path)['simulations'][1])


def _find(directory, pattern):
    return sorted(f for f in os.listdir(directory) if re.search(pattern, f))


# ============================================================
# STAGE 1: assemble obs + ensemble arrays, crop to INCA domain
# ============================================================

def assemble_arrays(run_dir, members, timestamp, stage1_dir):
    """
    Returns obs (INCA_NX, INCA_NY, 37) and sim_ensemble (INCA_NX, INCA_NY, 37, n_members).
    Results are cached as .npz files in stage1_dir.
    Returns (None, None) if observation files are missing.
    """
    npz_obs = os.path.join(stage1_dir, f"obs.{timestamp}.npz")
    npz_ens = os.path.join(stage1_dir, f"sim_ensemble.{timestamp}.npz")

    if os.path.exists(npz_obs) and os.path.exists(npz_ens):
        return np.load(npz_obs)['obs'], np.load(npz_ens)['sim_ensemble']

    member00_dir = os.path.join(run_dir, "MEMBER.00")
    obs_files    = _find(member00_dir, r"npc\.observations\.")
    fobs_files   = _find(member00_dir, r"npc\.future\.observations\.")

    obs_match  = [f for f in obs_files  if re.search(timestamp, f)]
    fobs_match = [f for f in fobs_files if re.search(timestamp, f)]

    if not obs_match or not fobs_match:
        print(f"  Obs/future-obs missing for {timestamp}, skipping")
        return None, None

    obs_arr  = _load_obs(os.path.join(member00_dir, obs_match[0]))
    fobs_arr = _load_future_obs(os.path.join(member00_dir, fobs_match[0]))

    # shape (710, 640, 37): t=0 is current obs, t=1..36 are future lead times
    obs_full = np.concatenate([obs_arr[:, :, 2:3], fobs_arr], axis=2)
    obs      = obs_full[X_SL, Y_SL, :].astype(np.float32)   # (501, 371, 37)

    n_members    = len(members)
    sim_ensemble = np.full((INCA_NX, INCA_NY, N_LEAD, n_members), np.nan, dtype=np.float32)

    for idx, m in enumerate(members):
        member_dir = os.path.join(run_dir, f"MEMBER.{m:02d}")
        sim_files  = _find(member_dir, r"npc\.simulations\.")
        sim_match  = [f for f in sim_files if re.search(rf"npc\.simulations.*{timestamp}.*\.rda$", f)]
        if not sim_match:
            print(f"  Simulation missing for member {m}, timestamp {timestamp}")
            continue
        sim_arr = _load_sim(os.path.join(member_dir, sim_match[0]))
        sim_full = np.concatenate([obs_arr[:, :, 2:3], sim_arr], axis=2)
        sim_ensemble[:, :, :, idx] = sim_full[X_SL, Y_SL, :]

    np.savez_compressed(npz_obs, obs=obs)
    np.savez_compressed(npz_ens, sim_ensemble=sim_ensemble)
    return obs, sim_ensemble


# ============================================================
# STAGE 2: exceedance probability maps
# ============================================================

def compute_exceedance(obs, sim_ensemble, threshold):
    """
    Threshold obs and ensemble, compute ensemble-mean exceedance probability.

    Returns
    -------
    obs_a    : (501, 371, 37) binary float32
    sim_prob : (501, 371, 37) ensemble-mean exceedance probability
    """
    obs_a    = (obs > threshold).astype(np.float32)
    ens_a    = (sim_ensemble > threshold).astype(np.float32)
    sim_prob = np.nanmean(ens_a, axis=3)
    return obs_a, sim_prob


# ============================================================
# STAGE 3: contingency tables + Brier score
# ============================================================

def contingency_and_brier(obs_a, sim_prob):
    """
    For each probability threshold and lead time compute:
      hits   — POD: P(forecast > pr | obs = 1)
      nohits — FAR: P(forecast > pr | obs = 0)
      fbb0   — frequency of forecasts in bin [pr, pr+0.1]  (denominator = N_PIXELS)
      orf    — observed relative frequency within that bin

    Brier score is computed once per lead time (independent of pr).

    Returns dict with arrays:
      hits, nohits, fbb0, orf : (n_thresholds, n_lead)
      brier                   : (n_lead,)
    """
    n_thr  = len(PROB_THRESHOLDS)
    hits   = np.full((n_thr, N_LEAD), np.nan)
    nohits = np.full((n_thr, N_LEAD), np.nan)
    fbb0   = np.full((n_thr, N_LEAD), np.nan)
    orf    = np.full((n_thr, N_LEAD), np.nan)
    brier  = np.full(N_LEAD, np.nan)

    for t in range(N_LEAD):
        o = obs_a[:, :, t].ravel()
        p = sim_prob[:, :, t].ravel()

        # Brier score: mean over all pixels (NaN pixels skipped, denominator = N_PIXELS)
        diff_sq = (p - o) ** 2
        brier[t] = np.nansum(diff_sq) / N_PIXELS

        for l, pr in enumerate(PROB_THRESHOLDS):
            aa = np.nansum((o == 1) & (p >  pr))   # hits
            bb = np.nansum((o == 1) & (p <= pr))   # misses
            cc = np.nansum((o == 0) & (p >  pr))   # false alarms
            dd = np.nansum((o == 0) & (p <= pr))   # correct negatives

            hits[l, t]   = aa / (aa + bb) if (aa + bb) > 0 else np.nan
            nohits[l, t] = cc / (cc + dd) if (cc + dd) > 0 else np.nan

            # pixels in probability bin [pr, pr+0.1] — matches R's >= and <=
            in_bin = ~np.isnan(p) & (p >= pr) & (p <= pr + 0.1)
            fbb0[l, t] = np.sum(in_bin) / N_PIXELS
            orf[l, t]  = np.nanmean(o[in_bin]) if np.any(in_bin) else np.nan

    return dict(hits=hits, nohits=nohits, fbb0=fbb0, orf=orf, brier=brier)


# ============================================================
# STAGE 4: rank histogram (Talagrand diagram)
# ============================================================

def update_rank_histogram(rank_hist, obs, sim_ensemble):
    """
    For each pixel and lead time rank the observation within the ensemble.
    rank = #{members obs exceeds} + 1, range 1..n_bins.
    Accumulates counts into rank_hist (n_bins, n_lead).
    """
    n_members = sim_ensemble.shape[3]
    n_bins    = n_members + 1

    for t in range(N_LEAD):
        obs_vec = obs[:, :, t].ravel()
        ens_mat = sim_ensemble[:, :, t, :].reshape(-1, n_members)
        valid   = ~np.isnan(obs_vec) & ~np.any(np.isnan(ens_mat), axis=1)
        if not np.any(valid):
            continue
        o     = obs_vec[valid]
        e     = ens_mat[valid]
        ranks = np.sum(o[:, None] > e, axis=1).astype(int) + 1
        rank_hist[:, t] += np.bincount(ranks, minlength=n_bins + 1)[1:]

    return rank_hist


# ============================================================
# STAGE 5: CRPS and tail-weighted CRPS
# ============================================================

def _crps_per_pixel(obs_vec, ens_mat):
    """
    obs_vec : (n_valid,)
    ens_mat : (n_valid, n_members) — no NaNs, pre-filtered

    Uses the sorted-member identity to avoid an O(m^2) pairwise tensor:
        sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - m - 1) * x_(k)  [k 1-indexed, sorted]
    so  E|X-X'|/2 = (1/m^2) * sum_k (2k-m-1) * x_(k)
    """
    n_members   = ens_mat.shape[1]
    term1       = np.mean(np.abs(ens_mat - obs_vec[:, None]), axis=1)
    ens_sorted  = np.sort(ens_mat, axis=1)
    k           = np.arange(1, n_members + 1, dtype=np.float64)
    weights     = 2.0 * k - n_members - 1.0
    term2       = ens_sorted @ weights / (n_members ** 2)
    return term1 - term2


def compute_crps(obs, sim_ensemble):
    """
    Standard ensemble CRPS averaged over valid pixels per lead time.
    Returns (N_LEAD,).
    """
    n_members = sim_ensemble.shape[3]
    crps_out  = np.full(N_LEAD, np.nan)
    for t in range(N_LEAD):
        obs_vec = obs[:, :, t].ravel()
        ens_mat = sim_ensemble[:, :, t, :].reshape(-1, n_members)
        valid   = ~np.isnan(obs_vec) & ~np.any(np.isnan(ens_mat), axis=1)
        if not np.any(valid):
            continue
        crps_out[t] = np.mean(_crps_per_pixel(obs_vec[valid], ens_mat[valid]))
    return crps_out


def compute_tw_crps(obs, sim_ensemble, threshold):
    """
    Tail-weighted CRPS: CRPS applied to max(x-v, 0) and max(y-v, 0).
    Scores only the part of the distribution above `threshold`, so large
    errors in the upper tail are penalised more heavily than by standard CRPS.
    Returns (N_LEAD,).
    """
    obs_c     = np.maximum(obs - threshold, 0.0)
    ens_c     = np.maximum(sim_ensemble - threshold, 0.0)
    n_members = sim_ensemble.shape[3]
    tw_out    = np.full(N_LEAD, np.nan)
    for t in range(N_LEAD):
        obs_vec = obs_c[:, :, t].ravel()
        ens_mat = ens_c[:, :, t, :].reshape(-1, n_members)
        valid   = ~np.isnan(obs_vec) & ~np.any(np.isnan(ens_mat), axis=1)
        if not np.any(valid):
            continue
        tw_out[t] = np.mean(_crps_per_pixel(obs_vec[valid], ens_mat[valid]))
    return tw_out


# ============================================================
# NetCDF OUTPUT
# ============================================================

def save_netcdf(results, members, output_dir, prefix, thresholds):
    path         = os.path.join(output_dir, f"{prefix}.nc")
    n_prob_thr   = len(PROB_THRESHOLDS)
    n_precip_thr = len(thresholds)
    n_bins       = len(members) + 1

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lead_time",        N_LEAD)
        ds.createDimension("prob_threshold",   n_prob_thr)
        ds.createDimension("precip_threshold", n_precip_thr)
        ds.createDimension("rank_bin",         n_bins)

        lt       = ds.createVariable("lead_time", "i4", ("lead_time",))
        lt[:]    = np.arange(N_LEAD)
        lt.units = "5-minute steps from analysis time"

        pt    = ds.createVariable("prob_threshold", "f4", ("prob_threshold",))
        pt[:] = PROB_THRESHOLDS.astype(np.float32)

        prt       = ds.createVariable("precip_threshold", "f4", ("precip_threshold",))
        prt[:]    = np.array(thresholds, dtype=np.float32)
        prt.units = "mm/10min"

        rb    = ds.createVariable("rank_bin", "i4", ("rank_bin",))
        rb[:] = np.arange(1, n_bins + 1)

        for name, long_name in [
            ("hits",   "Probability of Detection (POD) per probability threshold"),
            ("nohits", "False Alarm Rate per probability threshold"),
            ("fbb0",   "Forecast frequency in probability bin"),
            ("orf",    "Observed relative frequency in probability bin"),
        ]:
            v = ds.createVariable(name, "f4", ("precip_threshold", "prob_threshold", "lead_time"), fill_value=np.nan)
            v[:] = results[name].astype(np.float32)
            v.long_name = long_name

        bs           = ds.createVariable("brier_score", "f4", ("precip_threshold", "lead_time"), fill_value=np.nan)
        bs[:]        = results["brier"].astype(np.float32)
        bs.long_name = "Brier Score"

        cr           = ds.createVariable("crps",    "f4", ("lead_time",), fill_value=np.nan)
        cr[:]        = results["crps"].astype(np.float32)
        cr.long_name = "Continuous Ranked Probability Score"

        tw           = ds.createVariable("tw_crps", "f4", ("precip_threshold", "lead_time"), fill_value=np.nan)
        tw[:]        = results["tw_crps"].astype(np.float32)
        tw.long_name = "Tail-weighted CRPS (censored above precip_threshold)"

        rh           = ds.createVariable("rank_histogram", "i8", ("rank_bin", "lead_time"))
        rh[:]        = results["rank_hist"]
        rh.long_name = "Rank histogram counts (Talagrand diagram)"

        ds.n_members    = len(members)
        ds.n_timestamps = int(results["n_timestamps"])
        ds.inca_nx      = INCA_NX
        ds.inca_ny      = INCA_NY

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Probabilistic skill score verification for precipitation nowcasting ensembles."
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Root directory containing MEMBER.00, MEMBER.01, ... subdirs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where output NetCDF and stage1 cache are written")
    parser.add_argument("--csv",        required=True,
                        help="CSV with a 'Timestamps' column")
    parser.add_argument("--threshold",  type=float, nargs="+",
                        default=[1/6, 2/6, 5/6, 10/6],
                        metavar="THR",
                        help="One or more precipitation thresholds in mm/10min "
                             "(default: 1/6, 2/6, 5/6, 10/6 → 1, 2, 5, 10 mm/hr)")
    parser.add_argument("--members",    type=int, nargs=2, default=[0, 10],
                        metavar=("FIRST", "LAST"),
                        help="Inclusive range of member indices (default: 0 10)")
    parser.add_argument("--prefix",     default="npc1.prob",
                        help="Output NetCDF filename prefix (default: npc1.prob)")
    parser.add_argument("--stage1-dir", default=None,
                        help="Directory for intermediate .npz cache (default: output-dir/stage1)")
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    RUN_DIR    = args.run_dir
    OUTPUT_DIR = args.output_dir
    CSV_PATH   = args.csv
    THRESHOLDS = args.threshold
    MEMBERS    = list(range(args.members[0], args.members[1] + 1))
    PREFIX     = args.prefix
    STAGE1_DIR = args.stage1_dir or os.path.join(OUTPUT_DIR, "stage1")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STAGE1_DIR, exist_ok=True)

    timestamps = pd.read_csv(CSV_PATH)["Timestamps"].astype(str).tolist()
    print(f"Loaded {len(timestamps)} timestamps from {CSV_PATH}\n")

    n_prob_thr   = len(PROB_THRESHOLDS)
    n_precip_thr = len(THRESHOLDS)
    n_bins       = len(MEMBERS) + 1

    # Accumulators: leading axis is precipitation threshold index
    hits_sum   = np.zeros((n_precip_thr, n_prob_thr, N_LEAD))
    nohits_sum = np.zeros((n_precip_thr, n_prob_thr, N_LEAD))
    fbb0_sum   = np.zeros((n_precip_thr, n_prob_thr, N_LEAD))
    orf_sum    = np.zeros((n_precip_thr, n_prob_thr, N_LEAD))
    brier_sum  = np.zeros((n_precip_thr, N_LEAD))
    hits_cnt   = np.zeros((n_precip_thr, n_prob_thr, N_LEAD), dtype=int)
    nohits_cnt = np.zeros((n_precip_thr, n_prob_thr, N_LEAD), dtype=int)
    fbb0_cnt   = np.zeros((n_precip_thr, n_prob_thr, N_LEAD), dtype=int)
    orf_cnt    = np.zeros((n_precip_thr, n_prob_thr, N_LEAD), dtype=int)
    brier_cnt    = np.zeros((n_precip_thr, N_LEAD), dtype=int)
    crps_sum     = np.zeros(N_LEAD)
    crps_cnt     = np.zeros(N_LEAD, dtype=int)
    tw_crps_sum  = np.zeros((n_precip_thr, N_LEAD))
    tw_crps_cnt  = np.zeros((n_precip_thr, N_LEAD), dtype=int)
    rank_hist    = np.zeros((n_bins, N_LEAD), dtype=np.int64)
    n_valid      = 0

    for k, timestamp in enumerate(timestamps):
        print(f"[{k+1}/{len(timestamps)}] {timestamp}")

        # Stage 1
        obs, sim_ensemble = assemble_arrays(RUN_DIR, MEMBERS, timestamp, STAGE1_DIR)
        if obs is None:
            continue

        # Stage 5a: CRPS (threshold-independent)
        crps_scores = compute_crps(obs, sim_ensemble)
        m = ~np.isnan(crps_scores)
        crps_sum[m] += crps_scores[m]
        crps_cnt[m] += 1

        # Stages 2–3 + 5b: repeat for each precipitation threshold
        for i, thr in enumerate(THRESHOLDS):
            obs_a, sim_prob = compute_exceedance(obs, sim_ensemble, thr)
            scores = contingency_and_brier(obs_a, sim_prob)

            for name, s, c in [
                ("hits",   hits_sum,   hits_cnt),
                ("nohits", nohits_sum, nohits_cnt),
                ("fbb0",   fbb0_sum,   fbb0_cnt),
                ("orf",    orf_sum,    orf_cnt),
            ]:
                m = ~np.isnan(scores[name])
                s[i][m] += scores[name][m]
                c[i][m] += 1

            m = ~np.isnan(scores["brier"])
            brier_sum[i][m] += scores["brier"][m]
            brier_cnt[i][m] += 1

            tw = compute_tw_crps(obs, sim_ensemble, thr)
            m  = ~np.isnan(tw)
            tw_crps_sum[i][m] += tw[m]
            tw_crps_cnt[i][m] += 1

        # Stage 4
        rank_hist = update_rank_histogram(rank_hist, obs, sim_ensemble)
        n_valid  += 1

    if n_valid == 0:
        print("No valid timestamps processed.")
        raise SystemExit(1)

    results = dict(
        hits      = np.where(hits_cnt     > 0, hits_sum     / hits_cnt,     np.nan),
        nohits    = np.where(nohits_cnt   > 0, nohits_sum   / nohits_cnt,   np.nan),
        fbb0      = np.where(fbb0_cnt     > 0, fbb0_sum     / fbb0_cnt,     np.nan),
        orf       = np.where(orf_cnt      > 0, orf_sum      / orf_cnt,      np.nan),
        brier     = np.where(brier_cnt    > 0, brier_sum    / brier_cnt,    np.nan),
        crps      = np.where(crps_cnt     > 0, crps_sum     / crps_cnt,     np.nan),
        tw_crps   = np.where(tw_crps_cnt  > 0, tw_crps_sum  / tw_crps_cnt,  np.nan),
        rank_hist = rank_hist,
        n_timestamps = n_valid,
    )

    save_netcdf(results, MEMBERS, OUTPUT_DIR, PREFIX, THRESHOLDS)
