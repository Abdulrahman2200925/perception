#!/usr/bin/env python3
"""
READ-ONLY audit of SUSTechPOINTS labels: completeness, class distribution,
points-per-box (oriented 3D box containment test), and box-size sanity.

Reads .pcd clouds + .json labels, computes, prints results. Writes NOTHING.
"""
import json
import math
import os
import sys

import numpy as np

DATA_DIR = os.path.expanduser(
    "~/workspace/Eng.ros/perception/tools/SUSTechPOINTS/data"
)
SCENES = ["outdoor_1631", "outdoor_1634", "outdoor_1636", "outdoor_1638"]
FRAMES = [f"{i:06d}" for i in range(12)]
EXPECTED_CLASSES = {"Car", "Pedestrian", "Cyclist"}
SPARSE_THRESHOLD = 10

# Rough real-world dimension ranges (meters) for size sanity. scale = (x,y,z).
# We compare footprint (larger vs smaller of x,y) + height, orientation-agnostic.
SIZE_RANGES = {
    "Car":        {"len": (3.0, 5.5), "wid": (1.4, 2.3), "hgt": (1.2, 2.2)},
    "Pedestrian": {"len": (0.2, 1.3), "wid": (0.2, 1.3), "hgt": (1.2, 2.1)},
    "Cyclist":    {"len": (1.0, 2.2), "wid": (0.3, 1.2), "hgt": (1.2, 2.1)},
}


def load_pcd_xyz(path):
    """Load x,y,z from an ASCII PCD (header ends at 'DATA ascii')."""
    pts = []
    with open(path) as f:
        in_data = False
        for line in f:
            if not in_data:
                if line.startswith("DATA"):
                    in_data = True
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.asarray(pts, dtype=np.float64)


def points_in_box(pts, center, scale, yaw):
    """Count points inside an oriented box. yaw about +z."""
    if pts.shape[0] == 0:
        return 0
    rel = pts - center  # (N,3)
    c, s = math.cos(-yaw), math.sin(-yaw)  # rotate into box-local frame
    lx = c * rel[:, 0] - s * rel[:, 1]
    ly = s * rel[:, 0] + c * rel[:, 1]
    lz = rel[:, 2]
    hx, hy, hz = scale[0] / 2.0, scale[1] / 2.0, scale[2] / 2.0
    inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)
    return int(np.count_nonzero(inside))


def size_flag(cls, scale):
    """Return a reason string if box dims are implausible, else ''."""
    sx, sy, sz = scale
    if min(sx, sy, sz) <= 0.05:
        return "near-zero scale"
    r = SIZE_RANGES.get(cls)
    if r is None:
        return ""
    length = max(sx, sy)   # longer footprint axis
    width = min(sx, sy)
    height = sz
    issues = []
    if not (r["len"][0] <= length <= r["len"][1]):
        issues.append(f"len {length:.2f} not in {r['len']}")
    if not (r["wid"][0] <= width <= r["wid"][1]):
        issues.append(f"wid {width:.2f} not in {r['wid']}")
    if not (r["hgt"][0] <= height <= r["hgt"][1]):
        issues.append(f"hgt {height:.2f} not in {r['hgt']}")
    return "; ".join(issues)


def main():
    rows = []           # every box
    completeness = {}   # (scene, frame) -> nboxes or None if missing
    class_counts = {}
    missing = []

    for scene in SCENES:
        for frame in FRAMES:
            lbl = os.path.join(DATA_DIR, scene, "label", f"{frame}.json")
            pcd = os.path.join(DATA_DIR, scene, "lidar", f"{frame}.pcd")
            if not os.path.exists(lbl):
                completeness[(scene, frame)] = None
                missing.append((scene, frame))
                continue
            with open(lbl) as f:
                boxes = json.load(f)
            completeness[(scene, frame)] = len(boxes)
            pts = load_pcd_xyz(pcd) if os.path.exists(pcd) else np.empty((0, 3))
            for b in boxes:
                cls = b.get("obj_type", "MISSING_TYPE")
                class_counts[cls] = class_counts.get(cls, 0) + 1
                psr = b["psr"]
                center = np.array([psr["position"]["x"],
                                   psr["position"]["y"],
                                   psr["position"]["z"]])
                scale = np.array([psr["scale"]["x"],
                                  psr["scale"]["y"],
                                  psr["scale"]["z"]])
                yaw = psr["rotation"]["z"]
                n_in = points_in_box(pts, center, scale, yaw)
                rng = float(np.linalg.norm(center[:2]))  # ground-plane range
                rows.append({
                    "scene": scene, "frame": frame, "cls": cls,
                    "n": n_in, "scale": scale, "range": rng,
                    "obj_id": b.get("obj_id", "?"),
                    "size_issue": size_flag(cls, scale),
                })

    # ---- Completeness ----
    print("=" * 70)
    print("B. COMPLETENESS (frame -> #boxes; MISSING flagged)")
    for scene in SCENES:
        counts = [completeness[(scene, f)] for f in FRAMES]
        cells = ["MISS" if c is None else str(c) for c in counts]
        total = sum(c for c in counts if c is not None)
        print(f"  {scene}: " + " ".join(f"{f[-2:]}:{c}"
              for f, c in zip(FRAMES, cells)) + f"   (total {total})")
    print(f"  MISSING label files: {missing if missing else 'NONE'}")

    # ---- Class distribution ----
    print("=" * 70)
    print("C. CLASS DISTRIBUTION (all 48 frames)")
    total_boxes = sum(class_counts.values())
    for cls in sorted(class_counts):
        tag = "" if cls in EXPECTED_CLASSES else "  <-- UNEXPECTED"
        print(f"  {cls:16s} {class_counts[cls]:4d}{tag}")
    print(f"  TOTAL boxes: {total_boxes}")

    # ---- Points-per-box ----
    ns = np.array([r["n"] for r in rows])
    print("=" * 70)
    print("D. POINTS-PER-BOX")
    print(f"  boxes: {len(ns)}  min={ns.min()}  "
          f"median={int(np.median(ns))}  max={ns.max()}  mean={ns.mean():.1f}")
    buckets = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]
    labels = ["0", "1-4", "5-9", "10-19", "20-49", "50+"]
    print("  histogram:")
    for (lo, hi), lab in zip(buckets, labels):
        c = int(np.count_nonzero((ns >= lo) & (ns <= hi)))
        bar = "#" * c
        print(f"    {lab:6s} {c:4d} {bar}")

    sparse = sorted([r for r in rows if r["n"] < SPARSE_THRESHOLD],
                    key=lambda r: (r["n"], r["range"]))
    print(f"\n  BOXES WITH < {SPARSE_THRESHOLD} POINTS: {len(sparse)}")
    print(f"  {'scene':13s} {'frm':4s} {'id':3s} {'class':11s} "
          f"{'pts':>4s} {'range_m':>8s}  dims(LxWxH)")
    for r in sparse:
        sx, sy, sz = r["scale"]
        L, W = max(sx, sy), min(sx, sy)
        print(f"  {r['scene']:13s} {r['frame'][-2:]:4s} {r['obj_id']:3s} "
              f"{r['cls']:11s} {r['n']:>4d} {r['range']:>8.1f}  "
              f"{L:.2f}x{W:.2f}x{sz:.2f}")

    # sparse vs distance correlation
    if len(ns) > 2:
        rr = np.array([r["range"] for r in rows])
        corr = np.corrcoef(rr, ns)[0, 1]
        far = rr > np.median(rr)
        print(f"\n  range vs points corr = {corr:.2f} "
              f"(negative => farther boxes are sparser)")
        print(f"  median points  near (range<=median) = "
              f"{int(np.median(ns[~far]))}   far = {int(np.median(ns[far]))}")

    # ---- Size sanity ----
    print("=" * 70)
    print("E. BOX-SIZE SANITY (implausible dims)")
    flagged = [r for r in rows if r["size_issue"]]
    if not flagged:
        print("  none flagged")
    for r in flagged:
        sx, sy, sz = r["scale"]
        L, W = max(sx, sy), min(sx, sy)
        print(f"  {r['scene']} {r['frame']} id{r['obj_id']} {r['cls']}: "
              f"{L:.2f}x{W:.2f}x{sz:.2f} -> {r['size_issue']}")

    print("=" * 70)
    print(f"SUMMARY: {total_boxes} boxes, {len(sparse)} with <{SPARSE_THRESHOLD} "
          f"points, {len(flagged)} size-flagged, {len(missing)} missing labels")

    # emit a machine-readable line for the report writer
    return {
        "rows": rows, "sparse": sparse, "flagged": flagged,
        "missing": missing, "class_counts": class_counts,
        "completeness": completeness, "ns": ns,
    }


if __name__ == "__main__":
    main()
