#!/usr/bin/env python3
"""
Livox Mid-360 CSV Evaluator  —  standalone, no ROS required.

Reads the CSV produced by lvx2_to_csv.py and prints:
  • No-return rate (reported separately, not mixed into stats)
  • Range stats on VALID points only
  • Per-range valid point counts (0-10 / 10-20 / 20-30 / 30-40 / >40 m)
  • Reflectivity histogram on VALID points only
  • Object-density estimate: avg valid pts landing on a 1.8m-wide target
    at 10 / 20 / 30 / 40 m — the key 3D-detection decision metric

Usage:
    python3 evaluate_csv.py Outdoor_sampledata.csv
"""

import math
import os
import sys
import time

import numpy as np


# ── tunables ─────────────────────────────────────────────────────────────────
CHUNK_ROWS     = 500_000   # rows to parse per numpy batch (≈50 MB RAM per chunk)
AZ_BIN_DEG     = 0.5       # azimuth histogram cell size (degrees)
EL_BIN_DEG     = 0.5       # elevation histogram cell size (degrees)
EL_RANGE       = (-30, 65) # generous elevation bounds to catch all Mid-360 returns

# Target object model for density estimation
OBJ_WIDTH_M    = 1.8       # car / cyclist horizontal width
OBJ_HEIGHT_M   = 1.5       # car height (bounding box vertical extent)
TARGET_DISTS   = [10, 20, 30, 40]


# ── helpers ───────────────────────────────────────────────────────────────────

def angular_size_deg(half_extent_m, dist_m):
    """Full angular width (degrees) of a target with half-extent at given distance."""
    return 2.0 * math.degrees(math.atan(half_extent_m / dist_m))


def load_chunk(f, max_rows, n_cols):
    """
    Read up to max_rows from an already-opened CSV file (past the header line).
    Returns int32 ndarray of shape (actual_rows, n_cols), or None when EOF.
    Handles the edge case where loadtxt returns a 1-D array for a single row.
    """
    try:
        chunk = np.loadtxt(f, delimiter=',', max_rows=max_rows, dtype=np.int32)
    except StopIteration:
        return None
    if chunk.size == 0:
        return None
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    return chunk


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f'Usage: python3 {os.path.basename(__file__)} <path/to/file.csv>')
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f'ERROR: file not found: {csv_path}')
        sys.exit(1)

    print(f'\n{"="*64}')
    print(f'  LIVOX MID-360 CSV EVALUATION')
    print(f'  File: {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB)')
    print(f'{"="*64}')

    # ── discover column layout ────────────────────────────────────────────
    with open(csv_path) as fh:
        col_names = fh.readline().strip().split(',')

    def col(name):
        try:
            return col_names.index(name)
        except ValueError:
            return None

    xi  = col('x');            yi  = col('y');  zi  = col('z')
    ri  = col('reflectivity'); fi  = col('frame_index')

    if None in (xi, yi, zi, ri):
        print(f'ERROR: CSV must have columns x,y,z,reflectivity. Found: {col_names}')
        sys.exit(1)

    n_cols = len(col_names)
    print(f'\n  Columns ({n_cols}): {col_names}')
    print(f'  frame_index column: {"present" if fi is not None else "absent (frames estimated)"}')

    # ── accumulators ──────────────────────────────────────────────────────
    total_rows  = 0
    valid_rows  = 0

    # Range histogram: 0-5, 5-10, 10-15, …, 45-50, >50 m  (12 bins)
    R_EDGES  = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, np.inf])
    r_hist   = np.zeros(len(R_EDGES) - 1, dtype=np.int64)

    r_min    =  np.inf
    r_max    = -np.inf
    r_sum    = 0.0
    r_sumsq  = 0.0

    # Reflectivity histogram: 8 equal bins over 0-255
    REFL_EDGES = np.linspace(0, 256, 9)   # [0,32,64,…,224,256]
    refl_hist  = np.zeros(8, dtype=np.int64)
    refl_sum   = 0.0
    refl_sumsq = 0.0
    refl_min   = 255
    refl_max   = 0

    # 2-D azimuth × elevation occupancy grid for angular-density estimate
    az_edges = np.arange(-180, 180 + AZ_BIN_DEG, AZ_BIN_DEG)
    el_edges = np.arange(EL_RANGE[0], EL_RANGE[1] + EL_BIN_DEG, EL_BIN_DEG)
    n_az     = len(az_edges) - 1    # 720 cells
    n_el     = len(el_edges) - 1    # 190 cells
    az_el_occ = np.zeros((n_az, n_el), dtype=bool)   # ever-seen mask

    # Per-frame valid-point counter (for avg pts/frame)
    frame_valid = {}   # frame_index (int) → valid_count (int)

    t0 = time.time()

    # ── streaming parse ───────────────────────────────────────────────────
    print(f'\n  Parsing CSV in {CHUNK_ROWS:,}-row chunks …')

    with open(csv_path) as fh:
        fh.readline()  # skip header

        chunk_n = 0
        while True:
            chunk = load_chunk(fh, CHUNK_ROWS, n_cols)
            if chunk is None:
                break

            total_rows += len(chunk)
            chunk_n   += 1

            x = chunk[:, xi].astype(np.int32)
            y = chunk[:, yi].astype(np.int32)
            z = chunk[:, zi].astype(np.int32)
            r = chunk[:, ri].astype(np.uint8)

            # ── validity mask: exclude exact (0,0,0) no-return points ────
            valid = ~((x == 0) & (y == 0) & (z == 0))
            valid_rows += int(valid.sum())

            xv = x[valid].astype(np.float64)
            yv = y[valid].astype(np.float64)
            zv = z[valid].astype(np.float64)
            rv = r[valid].astype(np.float64)

            if len(xv) == 0:
                continue

            # ── range in metres (mm → m) ──────────────────────────────────
            dist_m = np.sqrt(xv**2 + yv**2 + zv**2) * 1e-3

            r_min   = min(r_min, float(dist_m.min()))
            r_max   = max(r_max, float(dist_m.max()))
            r_sum  += float(dist_m.sum())
            r_sumsq += float((dist_m**2).sum())

            rh, _ = np.histogram(dist_m, bins=R_EDGES)
            r_hist += rh

            # ── reflectivity ──────────────────────────────────────────────
            refl_sum   += float(rv.sum())
            refl_sumsq += float((rv**2).sum())
            refl_min    = min(refl_min, int(rv.min()))
            refl_max    = max(refl_max, int(rv.max()))
            rh2, _ = np.histogram(rv, bins=REFL_EDGES)
            refl_hist  += rh2

            # ── azimuth / elevation for FOV density ───────────────────────
            az = np.degrees(np.arctan2(yv, xv))           # −180 to +180 °
            el = np.degrees(np.arctan2(zv, np.sqrt(xv**2 + yv**2)))

            az_idx = np.searchsorted(az_edges, az, side='right') - 1
            el_idx = np.searchsorted(el_edges, el, side='right') - 1

            in_grid = (az_idx >= 0) & (az_idx < n_az) & \
                      (el_idx >= 0) & (el_idx < n_el)
            az_el_occ[az_idx[in_grid], el_idx[in_grid]] = True

            # ── per-frame valid counts ────────────────────────────────────
            if fi is not None:
                frame_ids = chunk[:, fi][valid]
                uniq, cnts = np.unique(frame_ids, return_counts=True)
                for fid, cnt in zip(uniq.tolist(), cnts.tolist()):
                    frame_valid[fid] = frame_valid.get(fid, 0) + cnt

            if chunk_n % 10 == 0:
                elapsed = time.time() - t0
                pct = total_rows / (os.path.getsize(csv_path) / 80) * 100
                print(f'    chunk {chunk_n:3d}  rows={total_rows:10,}  '
                      f'valid={valid_rows:10,}  ({elapsed:.1f}s)')

    elapsed = time.time() - t0

    # ── derived quantities ────────────────────────────────────────────────
    n_valid = valid_rows
    n_invalid = total_rows - valid_rows
    no_ret_pct = 100.0 * n_invalid / total_rows if total_rows else 0.0

    n_frames = len(frame_valid) if frame_valid else max(1, round(total_rows / 10_000))
    avg_valid_per_frame = n_valid / n_frames

    r_mean = r_sum / n_valid if n_valid else 0.0
    r_var  = r_sumsq / n_valid - r_mean**2 if n_valid else 0.0
    r_std  = math.sqrt(max(r_var, 0.0))

    refl_mean = refl_sum / n_valid if n_valid else 0.0
    refl_var  = refl_sumsq / n_valid - refl_mean**2 if n_valid else 0.0
    refl_std  = math.sqrt(max(refl_var, 0.0))

    # Effective FOV from occupied angular cells
    occupied_cells   = int(az_el_occ.sum())
    eff_fov_sq_deg   = occupied_cells * AZ_BIN_DEG * EL_BIN_DEG
    # Angular density: valid pts per sq-degree per frame
    ang_density      = avg_valid_per_frame / eff_fov_sq_deg if eff_fov_sq_deg > 0 else 0.0

    # ── print report ─────────────────────────────────────────────────────
    W = 64
    sep = '─' * W

    print(f'\n{"="*W}')
    print(f'  POINT COUNT OVERVIEW')
    print(sep)
    print(f'  Total rows in CSV           : {total_rows:>12,}')
    print(f'  No-return (0,0,0) points    : {n_invalid:>12,}  ← {no_ret_pct:.1f}% of raw scan')
    print(f'  Valid returns               : {n_valid:>12,}  ← used for ALL stats below')
    print(f'  Frames                      : {n_frames:>12,}')
    print(f'  Avg valid pts / frame       : {avg_valid_per_frame:>12,.0f}')
    print(f'  (Parse time: {elapsed:.1f}s)')

    # ── range stats ───────────────────────────────────────────────────────
    print(f'\n{"="*W}')
    print(f'  RANGE STATISTICS  (valid points only, coordinates in metres)')
    print(sep)
    print(f'  Min  : {r_min:7.2f} m')
    print(f'  Max  : {r_max:7.2f} m')
    print(f'  Mean : {r_mean:7.2f} m')
    print(f'  Std  : {r_std:7.2f} m')

    # ── per-range counts with cumulative ─────────────────────────────────
    print(f'\n{"="*W}')
    print(f'  VALID POINT DENSITY BY RANGE  (cumulative shows reachable fraction)')
    print(sep)
    print(f'  {"Band":<12}  {"Count":>10}  {"% of valid":>10}  {"Cumul ≤ max":>12}  {"Cumul %":>8}')
    print(f'  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*12}  {"-"*8}')

    band_labels = [
        ' 0 – 5 m', ' 5 –10 m', '10 –15 m', '15 –20 m',
        '20 –25 m', '25 –30 m', '30 –35 m', '35 –40 m',
        '40 –45 m', '45 –50 m', '    >50 m',
    ]
    cumul = 0
    for i, cnt in enumerate(r_hist):
        cnt  = int(cnt)
        cumul += cnt
        pct   = 100.0 * cnt   / n_valid if n_valid else 0.0
        cpct  = 100.0 * cumul / n_valid if n_valid else 0.0
        label = band_labels[i] if i < len(band_labels) else f'bin {i}'
        print(f'  {label:<12}  {cnt:>10,}  {pct:>9.1f}%  {cumul:>12,}  {cpct:>7.1f}%')

    # Highlight the four key thresholds
    # Build cumulative at exactly 10/20/30/40 m
    print()
    for threshold in [10, 20, 30, 40]:
        idx    = int(threshold / 5)   # bin index for ≤ threshold m  (e.g. 10m → bins 0+1)
        c      = int(r_hist[:idx].sum())
        pct    = 100.0 * c / n_valid if n_valid else 0.0
        print(f'  ≤{threshold:2d} m  →  {c:>10,} valid pts  ({pct:.1f}%)')

    # ── reflectivity ──────────────────────────────────────────────────────
    print(f'\n{"="*W}')
    print(f'  REFLECTIVITY  (valid points only, 0 – 255 integer)')
    print(sep)
    print(f'  Min  : {refl_min}')
    print(f'  Max  : {refl_max}')
    print(f'  Mean : {refl_mean:.2f}')
    print(f'  Std  : {refl_std:.2f}')
    if refl_std < 10:
        print(f'  *** LOW STD — intensity channel may be collapsed / unusable ***')
    elif refl_std > 40:
        print(f'  ✓  High std → channel well-spread → usable for surface classification')
    else:
        print(f'  ✓  Moderate spread → usable for basic intensity-based segmentation')

    print(f'\n  8-bin histogram (0–255):')
    bar_max = max(refl_hist.max(), 1)
    for i in range(8):
        lo  = int(REFL_EDGES[i])
        hi  = int(REFL_EDGES[i + 1])
        cnt = int(refl_hist[i])
        pct = 100.0 * cnt / n_valid if n_valid else 0.0
        bar = '█' * int(38 * cnt / bar_max)
        print(f'  [{lo:3d}–{hi:3d}]  {bar:<38s}  {cnt:>10,}  ({pct:5.1f}%)')

    # ── angular density & object coverage ────────────────────────────────
    print(f'\n{"="*W}')
    print(f'  ANGULAR DENSITY  (basis for object-coverage estimate)')
    print(sep)
    print(f'  Effective FOV (occupied {AZ_BIN_DEG}°×{EL_BIN_DEG}° cells): '
          f'{occupied_cells:,} cells = {eff_fov_sq_deg:.0f} sq-deg')
    print(f'  Mid-360 nominal FOV: 360° × 59° = 21 240 sq-deg')
    print(f'  Angular density: {ang_density:.4f} valid pts / sq-deg / frame')
    print(f'  (= {ang_density * 1e4:.2f} × 10⁻⁴ pts/sq-deg/frame)')

    print(f'\n{"="*W}')
    print(f'  OBJECT-COVERAGE ESTIMATE')
    print(f'  Target: {OBJ_WIDTH_M}m wide × {OBJ_HEIGHT_M}m tall  '
          f'(approx. car / large pedestrian)')
    print(f'  Metric: avg VALID pts landing on target IN ONE FRAME')
    print(sep)

    hdr = f'  {"Dist":>6}  {"Az-width":>9}  {"El-height":>10}  '  \
          f'{"Obj area":>10}  {"Pts/frame":>11}  {"Verdict"}'
    print(hdr)
    print(f'  {"-"*6}  {"-"*9}  {"-"*10}  {"-"*10}  {"-"*11}  {"-"*18}')

    for d in TARGET_DISTS:
        az_w  = angular_size_deg(OBJ_WIDTH_M  / 2.0, d)   # degrees
        el_h  = angular_size_deg(OBJ_HEIGHT_M / 2.0, d)   # degrees
        area  = az_w * el_h                                # sq-deg
        pts   = ang_density * area                         # pts/frame

        if pts < 3:
            verdict = '✗  below detection threshold'
        elif pts < 8:
            verdict = '△  marginal (multi-frame needed)'
        elif pts < 25:
            verdict = '✓  detectable, coarse bbox'
        else:
            verdict = '✓✓ reliable, full 3D bbox'

        print(f'  {d:>4}m  {az_w:>8.2f}°  {el_h:>9.2f}°  '
              f'{area:>9.2f}°²  {pts:>10.1f}  {verdict}')

    print()
    print(f'  Note: "per frame" = {50}ms window (frame_dur from file header).')
    print(f'  Accumulating N frames multiplies pts/frame by N.')
    print(f'  Typical 3D detectors require ≥3 pts/frame for single-frame detection.')
    print(f'  PointPillars / CenterPoint benchmarks: ≥8 pts for reliable bbox.')

    print(f'\n{"="*W}\n')


if __name__ == '__main__':
    main()
