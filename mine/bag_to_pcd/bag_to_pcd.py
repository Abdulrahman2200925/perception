#!/usr/bin/env python3
"""
bag_to_pcd.py — Export diverse frames from ROS2 rosbags of sensor_msgs/PointCloud2
into ASCII .pcd files laid out for the SUSTechPOINTS 3D annotation tool.

Pipeline position:
    rosbag (Mid-360S PointCloud2)  ->  [THIS SCRIPT]  ->  <out>/<scene>/lidar/NNNNNN.pcd
                                                          ->  SUSTechPOINTS labeling

Design:
  * Bag reading / layout detection / point extraction are REUSED from the sibling
    converter mine/bag_to_bin/bag_to_bin.py (open_reader, resolve_topic,
    detect_layout, frame_arrays) — field offsets/datatypes come from PointField
    metadata, never hardcoded.
  * Per bag: pick `frames_per_bag` EVENLY-SPACED frames (stride = count // N) so the
    selection spreads across the whole recording (diverse dynamic content), not
    consecutive frames.
  * Output is ASCII PCD with minimal fields `x y z intensity`. Intensity is written
    RAW as an INTEGER (0-255), NOT divided by 255: SUSTechPOINTS's ASCII PCDLoader
    parses intensity with parseInt(), so [0,1] floats would all collapse to 0 (flat
    coloring). Raw ints give real visual contrast. This is a LABELING-VISUAL choice
    only; training-time intensity scaling is a separate, later concern.
  * (0,0,0) no-return points and NaN points are dropped (default on), like bag_to_bin.
  * One SUSTechPOINTS "scene" per bag: outdoor_20260718_1631 -> outdoor_1631.

Dependencies: Python 3.9+, numpy, rosbag2_py, sensor_msgs_py (ROS2 Humble).
Run with ROS sourced and conda deactivated:
    conda deactivate; source /opt/ros/humble/setup.bash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# --- reuse the bag-reading/layout/extract logic from the sibling bag_to_bin.py ---
_BAG_TO_BIN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bag_to_bin")
)
if _BAG_TO_BIN_DIR not in sys.path:
    sys.path.insert(0, _BAG_TO_BIN_DIR)

from bag_to_bin import (  # noqa: E402  (import after sys.path tweak, by design)
    open_reader,
    resolve_topic,
    detect_layout,
    frame_arrays,
)

from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def scene_name_for_bag(bag_dir: str) -> str:
    """outdoor_20260718_1631 -> outdoor_1631 (prefix + trailing time token)."""
    base = os.path.basename(bag_dir.rstrip("/"))
    parts = base.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[-1]}"
    return base


def count_topic_messages(bag_uri: str, topic: str) -> int:
    """First pass: count messages on `topic` (no deserialize — cheap)."""
    reader = open_reader(bag_uri)
    n = 0
    while reader.has_next():
        rtopic, _data, _t = reader.read_next()
        if rtopic == topic:
            n += 1
    return n


def select_indices(count: int, n_frames: int) -> tuple[int, list[int]]:
    """Evenly-spaced selection: stride = count // n_frames; indices 0, s, 2s, ...."""
    if count <= 0:
        return 0, []
    if count <= n_frames:
        # fewer messages than requested frames: take them all
        return 1, list(range(count))
    stride = count // n_frames
    return stride, [i * stride for i in range(n_frames)]


PCD_HEADER = (
    "# .PCD v0.7 - Point Cloud Data file format\n"
    "VERSION 0.7\n"
    "FIELDS x y z intensity\n"
    "SIZE 4 4 4 4\n"
    "TYPE F F F F\n"
    "COUNT 1 1 1 1\n"
    "WIDTH {n}\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS {n}\n"
    "DATA ascii\n"
)


def write_pcd_ascii(path: str, xyz: np.ndarray, intensity_int: np.ndarray) -> None:
    """Write an ASCII PCD: 'x y z intensity' lines, xyz as %.4f, intensity as int."""
    n = int(xyz.shape[0])
    lines = [PCD_HEADER.format(n=n)]
    # build the body in one pass; intensity already integer-valued
    for (x, y, z), inten in zip(xyz, intensity_int):
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} {int(inten)}\n")
    with open(path, "w") as fh:
        fh.write("".join(lines))


def process_frame(msg: PointCloud2, layout: dict, drop_zeros: bool):
    """Extract xyz+intensity, drop (0,0,0)/NaN. Returns (xyz_keep, intensity_raw_keep,
    n_in, n_out, i_min, i_max, i_mean) with intensity kept RAW (float, pre-round)."""
    xyz, intensity, _tag = frame_arrays(msg, layout)
    n_in = int(xyz.shape[0])

    zero_mask = (xyz == 0.0).all(axis=1)
    nan_mask = ~np.isfinite(xyz).all(axis=1) | ~np.isfinite(intensity)
    keep = ~nan_mask
    if drop_zeros:
        keep &= ~zero_mask

    xyz_k = xyz[keep]
    inten_k = intensity[keep]
    n_out = int(xyz_k.shape[0])

    i_min = float(inten_k.min()) if n_out else float("nan")
    i_max = float(inten_k.max()) if n_out else float("nan")
    i_mean = float(inten_k.mean()) if n_out else float("nan")
    return xyz_k, inten_k, n_in, n_out, i_min, i_max, i_mean


def export_bag(bag_dir: str, topic_arg: Optional[str], out_root: str,
               frames_per_bag: int, drop_zeros: bool, dry_run: bool) -> dict:
    bag_uri = os.path.abspath(bag_dir.rstrip("/"))
    scene = scene_name_for_bag(bag_uri)
    topic = resolve_topic(bag_uri, topic_arg)

    count = count_topic_messages(bag_uri, topic)
    stride, selected = select_indices(count, frames_per_bag)
    selected_set = set(selected)

    print(f"\n=== bag {os.path.basename(bag_uri)}  ->  scene '{scene}' ===")
    print(f"[cfg] topic={topic}  frame_count={count}  frames_per_bag={frames_per_bag}"
          f"  stride={stride}")
    print(f"[cfg] selected indices: {selected}")

    lidar_dir = os.path.join(out_root, scene, "lidar")
    if not dry_run:
        os.makedirs(lidar_dir, exist_ok=True)
        # SUSTechPOINTS's save handler writes labels into <scene>/label/ but does NOT
        # create that dir itself — pre-create it or the first "save" returns HTTP 500.
        os.makedirs(os.path.join(out_root, scene, "label"), exist_ok=True)

    manifest = {
        "source_bag": bag_uri,
        "scene": scene,
        "topic": topic,
        "frame_count": count,
        "frames_per_bag": frames_per_bag,
        "stride": stride,
        "selected_indices": selected,
        "intensity": "raw-integer (no /255)",
        "drop_zeros": bool(drop_zeros),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pcd_format": "ascii x y z intensity",
        "frames": [],
    }

    # second pass: extract & (optionally) write the selected frames, in order
    reader = open_reader(bag_uri)
    layout: Optional[dict] = None
    topic_msg_index = -1
    out_index = 0
    first_reported = False

    while reader.has_next():
        rtopic, data, t = reader.read_next()
        if rtopic != topic:
            continue
        topic_msg_index += 1
        if topic_msg_index not in selected_set:
            continue

        msg = deserialize_message(data, PointCloud2)
        if layout is None:
            layout = detect_layout(msg)
            print(f"[layout] point_step={layout['point_step']} "
                  f"frame_id={layout['frame_id']!r} "
                  f"intensity_field={layout['intensity_field']} "
                  f"({layout['intensity_datatype']})")

        xyz_k, inten_k, n_in, n_out, i_min, i_max, i_mean = process_frame(
            msg, layout, drop_zeros)

        if not first_reported:
            print(f"[first frame idx#{topic_msg_index}] points {n_in} -> {n_out} "
                  f"(after zero/NaN drop)")
            print(f"[first frame intensity RAW] min/max/mean = "
                  f"{i_min:.2f}/{i_max:.2f}/{i_mean:.2f}")
            first_reported = True

        inten_int = np.rint(inten_k).astype(np.int64)

        pcd_name = f"{out_index:06d}.pcd"
        if not dry_run:
            write_pcd_ascii(os.path.join(lidar_dir, pcd_name), xyz_k, inten_int)

        stamp = msg.header.stamp
        manifest["frames"].append({
            "pcd": pcd_name,
            "topic_msg_index": topic_msg_index,
            "bag_stamp_ns": int(t),
            "header_stamp_sec": int(stamp.sec),
            "header_stamp_nanosec": int(stamp.nanosec),
            "points_in": n_in,
            "points_out": n_out,
            "intensity_raw_min": round(i_min, 4),
            "intensity_raw_max": round(i_max, 4),
            "intensity_raw_mean": round(i_mean, 4),
        })
        out_index += 1

    if layout is None:
        eprint(f"ERROR: no messages found on topic '{topic}' in {bag_uri}.")
        return manifest

    print(f"[{'dry-run' if dry_run else 'wrote'}] {out_index} pcd for scene '{scene}'")

    if not dry_run:
        manifest_path = os.path.join(out_root, scene, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[manifest] {manifest_path}")

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export evenly-spaced frames from ROS2 bags into ASCII .pcd for "
                    "SUSTechPOINTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--bags-root", required=True,
                    help="Directory containing the bag folders (each a rosbag2 dir).")
    ap.add_argument("--out-root", required=True,
                    help="SUSTechPOINTS data dir; scenes are created under it.")
    ap.add_argument("--topic", default=None,
                    help="PointCloud2 topic. Default: autodetect the single one.")
    ap.add_argument("--frames-per-bag", type=int, default=12,
                    help="Number of evenly-spaced frames to export per bag.")
    ap.add_argument("--bag-glob", default="outdoor_*",
                    help="Glob (within --bags-root) selecting which bag dirs to export.")
    ap.add_argument("--drop-zeros", action=argparse.BooleanOptionalAction, default=True,
                    help="Drop (0,0,0) no-return points (default on).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Inspect and report; create no dirs/files.")
    args = ap.parse_args()

    if args.frames_per_bag < 1:
        eprint("ERROR: --frames-per-bag must be >= 1.")
        return 2

    bags_root = os.path.abspath(args.bags_root.rstrip("/"))
    out_root = os.path.abspath(args.out_root.rstrip("/"))
    if not os.path.isdir(bags_root):
        eprint(f"ERROR: --bags-root is not a directory: {bags_root}")
        return 2

    import glob
    bag_dirs = sorted(
        d for d in glob.glob(os.path.join(bags_root, args.bag_glob)) if os.path.isdir(d)
    )
    if not bag_dirs:
        eprint(f"ERROR: no bag dirs match {args.bag_glob!r} under {bags_root}")
        return 2

    print(f"[cfg] bags_root={bags_root}")
    print(f"[cfg] out_root={out_root}")
    print(f"[cfg] bags found ({len(bag_dirs)}): {[os.path.basename(b) for b in bag_dirs]}")
    print(f"[cfg] frames_per_bag={args.frames_per_bag}  drop_zeros={args.drop_zeros}  "
          f"dry_run={args.dry_run}")

    total = 0
    for bag_dir in bag_dirs:
        manifest = export_bag(
            bag_dir, args.topic, out_root, args.frames_per_bag,
            args.drop_zeros, args.dry_run,
        )
        total += len(manifest["frames"])

    print(f"\n[summary] {'would export' if args.dry_run else 'exported'} {total} pcd "
          f"across {len(bag_dirs)} scenes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
