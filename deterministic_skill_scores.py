#!/usr/bin/env python3
"""
Deterministic skill score verification for precipitation nowcasting ensemble members.

Metrics (per lead time):
  PCORR, RMSF, CORR, POD, FAR, CSI, CMAE, CMSE, CRMSE, FSS_1, FSS_17, FSS_33

Output: one NetCDF file per timestamp x member.

Usage:
    python deterministic_skill_scores.py \\
        --run-dir /path/to/RUN5/NOWCAST.SIMULATIONS/ \\
        --output-dir /path/to/output/ \\
        --csv /path/to/filtered_timestamps.csv \\
        [--threshold 0.1667] \\
        [--members 0 10] \\
        [--prefix npc.validation]
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import netCDF4 as nc
import rdata
from scipy.ndimage import uniform_filter

# ============================================================
# FIXED CONSTANTS (not user-configurable)
# ============================================================
FSS_SCALES    = [1, 17, 33]
METRIC_NAMES  = ["PCORR", "RMSF", "CORR", "POD", "FAR", "CSI",
                 "CMAE",  "CMSE", "CRMSE", "FSS_1", "FSS_17", "FSS_33"]

# ============================================================
# DOMAIN (matches R coordinate setup)
# ============================================================
NR_FRAMES     = 36
DIM_X, DIM_Y  = 710, 640
SWISS_CORNERS = (485, 75, 835, 295)
RADAR_CORNERS = (255, -160, 965, 480)
H             = 75

INCA_CORNERS  = (
    SWISS_CORNERS[0] - H,
    SWISS_CORNERS[1] - H,
    SWISS_CORNERS[2] + H,
    SWISS_CORNERS[3] + H,
)

# R 1-indexed positions; convert to Python 0-indexed slices below
_x1_r = INCA_CORNERS[0] - RADAR_CORNERS[0]
_x2_r = _x1_r + (INCA_CORNERS[2] - INCA_CORNERS[0])
_y1_r = INCA_CORNERS[1] - RADAR_CORNERS[1]
_y2_r = _y1_r + (INCA_CORNERS[3] - INCA_CORNERS[1])

# Python slices (R arr[a:b] inclusive → Python arr[a-1:b])
X_SL = slice(_x1_r - 1, _x2_r)
Y_SL = slice(_y1_r - 1, _y2_r)

# ============================================================
# FILE I/O
# ============================================================

def list_files(directory, pattern):
    files = sorted(f for f in os.listdir(directory) if re.search(pattern, f))
    return files


def match_timestamp(filenames, timestamp, r_start, r_end):
    """Return files where filename[r_start-1:r_end] == timestamp (R 1-indexed)."""
    return [f for f in filenames if f[r_start - 1:r_end] == timestamp]


def _load_rda(path):
    """Parse an .rda file and return a dict of Python objects."""
    parsed = rdata.parser.parse_file(path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        return rdata.conversion.convert(parsed)


def load_obs(path):
    """Load observations[[2]] from an .rda file — shape (710, 640, 3)."""
    data = _load_rda(path)
    return np.array(data['observations'][1])  # [[2]] in R = index 1 (0-based)


def load_future_obs(path):
    """Load observation.array.v from an .rda file — shape (710, 640, 36)."""
    data = _load_rda(path)
    return np.array(data['observation.array.v'])


def load_sim(path):
    """Load simulations[[2]] from an .rda file — shape (710, 640, 36)."""
    data = _load_rda(path)
    return np.array(data['simulations'][1])  # [[2]] in R = index 1 (0-based)


# ============================================================
# METRICS
# ============================================================

def _valid(obs, fct):
    return ~np.isnan(obs) & ~np.isnan(fct)


def corr(obs, fct):
    m = _valid(obs, fct)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(obs[m], fct[m])[0, 1])


def pcorr(obs, fct, threshold):
    """Correlation restricted to pixels where obs or fct exceeds threshold."""
    m = _valid(obs, fct) & ((obs >= threshold) | (fct >= threshold))
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(obs[m], fct[m])[0, 1])


def rmsf(obs, fct, threshold):
    """Root Mean Square Factor: exp(sqrt(mean(log(fct/obs)^2))) on rainy obs pixels."""
    m = _valid(obs, fct) & (obs > threshold) & (fct > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.exp(np.sqrt(np.mean(np.log(fct[m] / obs[m]) ** 2))))


def categorical(obs, fct, threshold):
    """POD, FAR, CSI."""
    m      = _valid(obs, fct)
    o_wet  = (obs >= threshold) & m
    f_wet  = (fct >= threshold) & m
    hits   = np.sum(o_wet  &  f_wet)
    misses = np.sum(o_wet  & ~f_wet)
    fa     = np.sum(~o_wet &  f_wet)
    pod    = hits / (hits + misses)  if (hits + misses) > 0 else np.nan
    far    = fa   / (hits + fa)      if (hits + fa)     > 0 else np.nan
    csi    = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else np.nan
    return float(pod), float(far), float(csi)


def conditional(obs, fct, threshold):
    """CMAE, CMSE, CRMSE where obs >= threshold."""
    m = _valid(obs, fct) & (obs >= threshold)
    if m.sum() == 0:
        return np.nan, np.nan, np.nan
    d    = fct[m] - obs[m]
    cmae  = float(np.mean(np.abs(d)))
    cmse  = float(np.mean(d ** 2))
    crmse = float(np.sqrt(cmse))
    return cmae, cmse, crmse


def fss(obs, fct, threshold, scale):
    """Fractions Skill Score for a given neighborhood size (pixels)."""
    o_bin = np.where(np.isnan(obs), 0.0, (obs >= threshold).astype(float))
    f_bin = np.where(np.isnan(fct), 0.0, (fct >= threshold).astype(float))
    o_frac = uniform_filter(o_bin, size=scale, mode="constant")
    f_frac = uniform_filter(f_bin, size=scale, mode="constant")
    mse = np.nanmean((o_frac - f_frac) ** 2)
    ref = np.nanmean(o_frac ** 2) + np.nanmean(f_frac ** 2)
    return float(1.0 - mse / ref) if ref > 0 else np.nan


def compute_skill_scores(obs_arr, fct_arr, threshold):
    """
    Compute all metrics for every lead time.

    Parameters
    ----------
    obs_arr, fct_arr : ndarray (x, y, n_times)
    threshold : float

    Returns
    -------
    scores : ndarray (n_times, 12)
    """
    n_times = obs_arr.shape[2]
    scores  = np.full((n_times, len(METRIC_NAMES)), np.nan)

    for t in range(n_times):
        o = obs_arr[:, :, t]
        f = fct_arr[:, :, t]

        pod, far, csi     = categorical(o, f, threshold)
        cmae, cmse, crmse = conditional(o, f, threshold)
        fss_vals          = [fss(o, f, threshold, s) for s in FSS_SCALES]

        scores[t] = [
            pcorr(o, f, threshold),
            rmsf(o, f, threshold),
            corr(o, f),
            pod, far, csi,
            cmae, cmse, crmse,
            *fss_vals,
        ]

    return scores


# ============================================================
# NetCDF OUTPUT
# ============================================================

def save_netcdf(scores, timestamp, member_index, output_dir, output_prefix, threshold):
    path = os.path.join(
        output_dir,
        f"{output_prefix}.{timestamp}_{member_index:03d}.nc"
    )
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lead_time", scores.shape[0])
        ds.createDimension("metric",    scores.shape[1])

        lt      = ds.createVariable("lead_time", "i4", ("lead_time",))
        lt[:]   = np.arange(scores.shape[0])
        lt.units = "5-minute steps from analysis time"

        sk = ds.createVariable(
            "skill_scores", "f4", ("lead_time", "metric"),
            fill_value=np.nan
        )
        sk[:]            = scores.astype(np.float32)
        sk.metric_names  = ", ".join(METRIC_NAMES)
        sk.threshold     = threshold
        sk.fss_scales    = str(FSS_SCALES)

        ds.timestamp     = timestamp
        ds.member        = member_index
        ds.output_prefix = output_prefix

    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic skill score verification for precipitation nowcasting ensemble members."
    )
    parser.add_argument("--run-dir",    required=True,
                        help="Root directory containing MEMBER.00, MEMBER.01, ... subdirs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where output NetCDF files are written")
    parser.add_argument("--csv",        required=True,
                        help="CSV file with a 'Timestamps' column listing timestamps to process")
    parser.add_argument("--threshold",  type=float, default=1/6,
                        help="Precipitation threshold in mm per 5 min (default: 1/6 ≈ 10 mm/h)")
    parser.add_argument("--members",    type=int, nargs=2, default=[0, 10], metavar=("FIRST", "LAST"),
                        help="Inclusive range of member indices to process (default: 0 10)")
    parser.add_argument("--prefix",     default="npc.validation",
                        help="Output filename prefix (default: npc.validation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RUN_DIR       = args.run_dir
    OUTPUT_DIR    = args.output_dir
    CSV_PATH      = args.csv
    THRESHOLD     = args.threshold
    MEMBERS       = range(args.members[0], args.members[1] + 1)
    OUTPUT_PREFIX = args.prefix

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamps = pd.read_csv(CSV_PATH)["Timestamps"].astype(str).tolist()
    print(f"Loaded {len(timestamps)} timestamps from {CSV_PATH}\n")

    # obs and fobs are the same across all members — load once from MEMBER.00
    member00_dir = os.path.join(RUN_DIR, "MEMBER.00")
    obs_files    = list_files(member00_dir, r"npc\.observations\.")
    fobs_files   = list_files(member00_dir, r"npc\.future\.observations\.")

    for member_index in MEMBERS:
        member_id  = f"MEMBER.{member_index:02d}"
        member_dir = os.path.join(RUN_DIR, member_id)

        sim_files = list_files(member_dir, r"npc\.simulations\.")

        for timestamp in timestamps:
            print(f"Processing {member_id} | {timestamp}")

            # Match files by timestamp substring (R positions: obs 18-26, fobs 25-33, sim 17-25)
            obs_match  = match_timestamp(obs_files,  timestamp, 18, 26)
            fobs_match = match_timestamp(fobs_files, timestamp, 25, 33)
            sim_match  = match_timestamp(sim_files,  timestamp, 17, 25)

            if not obs_match:
                print(f"  Observation file missing, skipping")
                continue
            if not fobs_match:
                print(f"  Future observation file missing, skipping")
                continue
            if not sim_match:
                print(f"  Simulation file missing, skipping")
                continue

            try:
                obs_array   = load_obs(os.path.join(member00_dir, obs_match[0]))
                obs_array_v = load_future_obs(os.path.join(member00_dir, fobs_match[0]))
                sim_array   = load_sim(os.path.join(member_dir,   sim_match[0]))
            except Exception as e:
                print(f"  Load error: {e}, skipping")
                continue

            # Assemble (710, 640, 37): t=0 is current obs, t=1..36 are future/forecast
            obs_full = np.concatenate([obs_array[:, :, 2:3], obs_array_v], axis=2)
            fct_full = np.concatenate([obs_array[:, :, 2:3], sim_array],   axis=2)

            # Crop to INCA domain
            obs_inca = obs_full[X_SL, Y_SL, :]
            fct_inca = fct_full[X_SL, Y_SL, :]
            print(f"  obs: {obs_inca.shape}, fct: {fct_inca.shape}")

            scores = compute_skill_scores(obs_inca, fct_inca, THRESHOLD)
            save_netcdf(scores, timestamp, member_index, OUTPUT_DIR, OUTPUT_PREFIX, THRESHOLD)
