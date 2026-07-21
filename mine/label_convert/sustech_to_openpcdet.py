#!/usr/bin/env python3
"""
sustech_to_openpcdet.py — Convert SUSTechPOINTS `psr` labels to OpenPCDet
custom_dataset lines `x y z dx dy dz heading category`.

Spec (from docs/reports/AUDIT_label_convert_plan_*):
  x=position.x, y=position.y, z=position.z   (identity, LiDAR frame, CENTER-based)
  dz = scale.z                               (height)
  CANONICALIZATION (the one risk — always applied):
     if scale.x >= scale.y:  dx,dy,heading = scale.x, scale.y, rotation.z
     else:                   dx,dy,heading = scale.y, scale.x, rotation.z + pi/2
     then wrap heading to (-pi, pi]
  category = obj_type verbatim ('Car'/'Pedestrian'); 7 numeric values + category
  (NO class id — OpenPCDet's prepare_data appends it). Cyclist: none present.
  Range assertion: box center within [-40.96,-40.96,-0.5, 40.96,40.96,3.0];
  outside → loud WARN with frame+obj_id (never silently dropped).

Stages:
  default (--dry-run) : print converted lines + stats, write NOTHING.
  --verify            : back-projection top-down plots for sample cars.
  --emit              : STAGE B — write .txt label files + MANIFEST_pairing.json.

Runs in the `sustechpoints` conda env (numpy + matplotlib). PCD reading is pure
python (no ROS). Reads JSON keys / PCD fields by NAME, never hardcoded offsets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

DATA_ROOT = os.path.expanduser(
    "~/workspace/Eng.ros/perception/tools/SUSTechPOINTS/data"
)
SCENES = ["outdoor_1631", "outdoor_1634", "outdoor_1636", "outdoor_1638"]
FRAMES = [f"{i:06d}" for i in range(12)]
POINT_CLOUD_RANGE = (-40.96, -40.96, -0.5, 40.96, 40.96, 3.0)  # xmin ymin zmin xmax ymax zmax

# Required visual-verification samples (scene, frame, obj_id, obj_type).
# Spans both annotation patterns: len-on-x (id45) vs len-on-y canonicalized (id48, 1631 id1).
VERIFY_SAMPLES = [
    ("outdoor_1634", "000005", "45", "Car"),   # len on scale.x, yaw ~ +90deg
    ("outdoor_1634", "000005", "48", "Car"),   # len on scale.y, yaw ~ 0   (canonicalized)
    ("outdoor_1631", "000000", "1",  "Car"),   # len on scale.y, yaw ~ +-180 (canonicalized)
    ("outdoor_1636", "000000", "1",  "Car"),   # normal len-on-x case
    ("outdoor_1638", "000000", "1",  "Car"),   # extra clearly-oriented car
]


def wrap_to_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi   # -> [-pi, pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def convert_box(psr: dict, obj_type: str):
    """Pure conversion: SUSTechPOINTS psr + type -> (x,y,z,dx,dy,dz,heading,category)."""
    pos, rot, sc = psr["position"], psr["rotation"], psr["scale"]
    x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
    dz = float(sc["z"])
    sx, sy = float(sc["x"]), float(sc["y"])
    if sx >= sy:
        dx, dy, heading = sx, sy, float(rot["z"])
        canon = False
    else:
        dx, dy, heading = sy, sx, float(rot["z"]) + math.pi / 2.0
        canon = True
    heading = wrap_to_pi(heading)
    return (x, y, z, dx, dy, dz, heading, obj_type), canon


def in_range(x: float, y: float, z: float) -> bool:
    xm, ym, zm, xM, yM, zM = POINT_CLOUD_RANGE
    return (xm <= x <= xM) and (ym <= y <= yM) and (zm <= z <= zM)


def load_labels(scene: str, frame: str):
    path = os.path.join(DATA_ROOT, scene, "label", f"{frame}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_pcd_xyz(path: str) -> np.ndarray:
    """Minimal ASCII-PCD reader (our SUSTech export: fields x y z intensity)."""
    pts = []
    with open(path) as f:
        in_data = False
        for line in f:
            if not in_data:
                if line.startswith("DATA"):
                    in_data = True
                continue
            p = line.split()
            if len(p) >= 3:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
    return np.asarray(pts, dtype=np.float64)


def box_corners_xy(x, y, dx, dy, heading):
    """4 top-down corners (CCW) of the oriented box footprint."""
    hx, hy = dx / 2.0, dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([x, y])


def points_in_box(pts, x, y, z, dx, dy, dz, heading):
    """Boolean mask of points inside the oriented box."""
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    rel = pts - np.array([x, y, z])
    c, s = math.cos(-heading), math.sin(-heading)
    lx = c * rel[:, 0] - s * rel[:, 1]
    ly = s * rel[:, 0] + c * rel[:, 1]
    lz = rel[:, 2]
    return (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(lz) <= dz / 2)


def fmt_line(conv) -> str:
    x, y, z, dx, dy, dz, heading, cat = conv
    return (f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} "
            f"{heading:.6f} {cat}")


# --------------------------------------------------------------------------- #
# STAGE A2 — dry-run over all frames
# --------------------------------------------------------------------------- #
def run_dry(show_samples=3):
    total = 0
    canon_count = 0
    per_scene = {}
    per_class = {}
    warnings = []
    canon_examples = []
    sample_prints = []

    for scene in SCENES:
        per_scene[scene] = 0
        for frame in FRAMES:
            boxes = load_labels(scene, frame)
            if boxes is None:
                warnings.append(f"MISSING label file: {scene}/{frame}")
                continue
            for b in boxes:
                conv, canon = convert_box(b["psr"], b["obj_type"])
                total += 1
                per_scene[scene] += 1
                per_class[conv[7]] = per_class.get(conv[7], 0) + 1
                if canon:
                    canon_count += 1
                    if len(canon_examples) < 6:
                        sc = b["psr"]["scale"]
                        canon_examples.append(
                            f"  {scene}/{frame} id{b.get('obj_id')} {conv[7]}: "
                            f"scale.x={sc['x']:.2f} scale.y={sc['y']:.2f} "
                            f"-> dx={conv[3]:.2f} dy={conv[4]:.2f} "
                            f"heading={math.degrees(conv[6]):+.1f}deg (yaw+90)")
                if not in_range(conv[0], conv[1], conv[2]):
                    warnings.append(
                        f"!! RANGE: {scene}/{frame} id{b.get('obj_id')} "
                        f"center=({conv[0]:.2f},{conv[1]:.2f},{conv[2]:.2f}) "
                        f"OUTSIDE {POINT_CLOUD_RANGE}")
                if (scene == "outdoor_1631" and frame == "000000"
                        and len(sample_prints) < show_samples):
                    sample_prints.append((b, conv, canon))

    print("=" * 70)
    print("STAGE A2 — DRY-RUN (nothing written)")
    print(f"TOTAL boxes converted: {total}")
    print(f"per-scene: " + "  ".join(f"{s}={per_scene[s]}" for s in SCENES))
    print(f"per-class: " + "  ".join(f"{k}={v}" for k, v in sorted(per_class.items())))
    print(f"canonicalized (scale.y>scale.x, heading+90): {canon_count} "
          f"({100*canon_count/total:.1f}%)")
    print("canonicalization examples:")
    for e in canon_examples:
        print(e)
    print(f"\nrange/other warnings: {len([w for w in warnings if 'RANGE' in w])} range, "
          f"{len([w for w in warnings if 'MISSING' in w])} missing")
    for w in warnings:
        print("  " + w)
    print("\n3 sample source-psr -> converted-line (outdoor_1631/000000):")
    for b, conv, canon in sample_prints:
        p = b["psr"]
        print(f"  SRC id{b.get('obj_id')} {b['obj_type']}: "
              f"pos=({p['position']['x']:.3f},{p['position']['y']:.3f},{p['position']['z']:.3f}) "
              f"scale=({p['scale']['x']:.3f},{p['scale']['y']:.3f},{p['scale']['z']:.3f}) "
              f"yaw={p['rotation']['z']:.4f}  canon={canon}")
        print(f"  OUT -> {fmt_line(conv)}")
    return dict(total=total, per_scene=per_scene, per_class=per_class,
                canon_count=canon_count, warnings=warnings)


# --------------------------------------------------------------------------- #
# STAGE A3 — visual back-projection verification
# --------------------------------------------------------------------------- #
def run_verify(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    results = []
    headings = {}
    for scene, frame, obj_id, obj_type in VERIFY_SAMPLES:
        boxes = load_labels(scene, frame)
        match = next((b for b in boxes
                      if str(b.get("obj_id")) == obj_id and b["obj_type"] == obj_type), None)
        if match is None:
            print(f"!! verify sample not found: {scene}/{frame} id{obj_id} {obj_type}")
            continue
        conv, canon = convert_box(match["psr"], obj_type)
        x, y, z, dx, dy, dz, heading, cat = conv

        pcd = os.path.join(DATA_ROOT, scene, "lidar", f"{frame}.pcd")
        pts = load_pcd_xyz(pcd)
        inside = points_in_box(pts, x, y, z, dx, dy, dz, heading)
        n_in = int(inside.sum())
        # cluster estimate: points inside a box inflated by 0.3 m each half-extent
        infl = points_in_box(pts, x, y, z, dx + 0.6, dy + 0.6, dz + 0.6, heading)
        n_cluster = int(infl.sum())
        capture = (100.0 * n_in / n_cluster) if n_cluster else float("nan")

        headings[(scene, frame, obj_id)] = math.degrees(heading)
        results.append(dict(scene=scene, frame=frame, obj_id=obj_id, cat=cat,
                            dx=dx, dy=dy, heading_deg=math.degrees(heading),
                            n_in=n_in, n_cluster=n_cluster, capture=capture, canon=canon))

        # ---- top-down plot ----
        fig, ax = plt.subplots(figsize=(8, 8))
        # local window around the box
        m = max(dx, dy) * 2.5 + 3
        sel = ((np.abs(pts[:, 0] - x) < m) & (np.abs(pts[:, 1] - y) < m))
        near = pts[sel]
        ax.scatter(near[:, 0], near[:, 1], s=2, c="#bbbbbb", label="cloud")
        ins = pts[inside]
        ax.scatter(ins[:, 0], ins[:, 1], s=6, c="#d1495b", label=f"in-box ({n_in})")
        corners = box_corners_xy(x, y, dx, dy, heading)
        poly = np.vstack([corners, corners[0]])
        ax.plot(poly[:, 0], poly[:, 1], "-", c="#0b6e4f", lw=2, label="box")
        # heading arrow along +local-x (the dx/length axis)
        ax.arrow(x, y, math.cos(heading) * dx / 2, math.sin(heading) * dx / 2,
                 head_width=0.3, head_length=0.4, fc="#1d3557", ec="#1d3557",
                 length_includes_head=True, zorder=5)
        ax.set_aspect("equal")
        ax.set_title(f"{scene}/{frame} id{obj_id} {cat} | dx={dx:.2f} dy={dy:.2f} "
                     f"hdg={math.degrees(heading):+.1f}deg canon={canon}\n"
                     f"capture={capture:.1f}% ({n_in}/{n_cluster})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        out_png = os.path.join(out_dir, f"{scene}_{frame}_{obj_id}.png")
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[verify] {scene}/{frame} id{obj_id} {cat}: dx={dx:.2f} dy={dy:.2f} "
              f"dx>=dy={dx >= dy} hdg={math.degrees(heading):+.1f}deg "
              f"capture={capture:.1f}% ({n_in}/{n_cluster}) canon={canon} -> {out_png}")

    # id45 vs id48 parallelism check (same physical row -> arrows must be parallel)
    h45 = headings.get(("outdoor_1634", "000005", "45"))
    h48 = headings.get(("outdoor_1634", "000005", "48"))
    if h45 is not None and h48 is not None:
        diff = abs(((h45 - h48) + 180) % 360 - 180)   # 0 or 180 both = parallel axis
        parallel = (diff < 10) or (abs(diff - 180) < 10)
        print(f"\n[parallel-check] id45 hdg={h45:+.1f}deg  id48 hdg={h48:+.1f}deg  "
              f"axis-diff={min(diff, abs(diff-180)):.1f}deg  PARALLEL={parallel}")
    return results


# --------------------------------------------------------------------------- #
# STAGE B — emit label files + pairing manifest
# --------------------------------------------------------------------------- #
def run_emit(label_out_root):
    os.makedirs(label_out_root, exist_ok=True)
    pairing = {"created_utc": None, "point_cloud_range": list(POINT_CLOUD_RANGE),
               "note": "SUSTechPOINTS->OpenPCDet custom labels; heading canonicalized (dx>=dy).",
               "frames": []}
    from datetime import datetime, timezone
    pairing["created_utc"] = datetime.now(timezone.utc).isoformat()

    per_scene = {}
    per_class = {}
    total = 0
    files = 0
    for scene in SCENES:
        # provenance from the SUSTech manifest
        man_path = os.path.join(DATA_ROOT, scene, "manifest.json")
        man = json.load(open(man_path)) if os.path.exists(man_path) else {}
        src_bag = man.get("source_bag")
        frame_prov = {fr["pcd"].split(".")[0]: fr for fr in man.get("frames", [])}
        out_scene = os.path.join(label_out_root, scene)
        os.makedirs(out_scene, exist_ok=True)
        per_scene[scene] = 0
        for frame in FRAMES:
            boxes = load_labels(scene, frame)
            if boxes is None:
                continue
            lines = []
            for b in boxes:
                conv, _ = convert_box(b["psr"], b["obj_type"])
                if not in_range(conv[0], conv[1], conv[2]):
                    print(f"!! RANGE WARN {scene}/{frame} id{b.get('obj_id')} outside range")
                lines.append(fmt_line(conv))
                per_class[conv[7]] = per_class.get(conv[7], 0) + 1
                per_scene[scene] += 1
                total += 1
            out_txt = os.path.join(out_scene, f"{frame}.txt")
            with open(out_txt, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            files += 1
            prov = frame_prov.get(frame, {})
            repo_root = os.path.expanduser("~/workspace/Eng.ros/perception")
            bag_base = os.path.basename(src_bag) if src_bag else scene
            pairing["frames"].append({
                "txt_path": os.path.relpath(out_txt, repo_root),
                "scene": scene, "frame_id": frame,
                "source_bag": src_bag,
                "topic_msg_index": prov.get("topic_msg_index"),
                "pcd_path": f"tools/SUSTechPOINTS/data/{scene}/lidar/{frame}.pcd",
                "intended_bin_path": f"data/derived/bin/{bag_base}/{frame}.bin",
                "num_boxes": len(lines),
            })

    man_out = os.path.join(label_out_root, "MANIFEST_pairing.json")
    with open(man_out, "w") as f:
        json.dump(pairing, f, indent=2)

    print("=" * 70)
    print("STAGE B — EMITTED label files")
    print(f"files written: {files}")
    for s in SCENES:
        print(f"  {s}: {sum(1 for _ in FRAMES)} frames, {per_scene[s]} boxes")
    print(f"per-class: " + "  ".join(f"{k}={v}" for k, v in sorted(per_class.items())))
    print(f"TOTAL boxes: {total}")
    print(f"pairing manifest: {man_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--verify", action="store_true",
                    help="produce top-down back-projection plots for sample cars")
    ap.add_argument("--emit", action="store_true",
                    help="STAGE B: actually write .txt label files (requires approval)")
    ap.add_argument("--verify-out",
                    default=os.path.expanduser("~/workspace/Eng.ros/perception/data/derived/verify"))
    ap.add_argument("--label-out",
                    default=os.path.expanduser("~/workspace/Eng.ros/perception/data/derived/labels_openpcdet"))
    args = ap.parse_args()

    if args.emit:
        run_emit(args.label_out)
        return 0
    # Stage A: dry-run always; verify if asked.
    run_dry()
    if args.verify:
        print()
        run_verify(args.verify_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
