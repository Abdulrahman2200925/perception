#!/usr/bin/env python3
"""
bag_to_bin.py — Convert a ROS2 rosbag of sensor_msgs/PointCloud2 into KITTI-format
.bin frames for OpenPCDet training/labeling.

Pipeline position:
    rosbag (Mid-360S PointCloud2)  ->  [THIS SCRIPT]  ->  KITTI .bin  ->  labeling  ->  OpenPCDet

Output: one file per selected message, KITTI convention:
    float32[N, 4] = (x, y, z, intensity), flat little-endian, no header,
    named 000000.bin, 000001.bin, ...

Design rules (see docs/reports/PHASE3A_bag_format.md and PINNED.md):
  * Field offsets/datatypes are read from each message's PointField metadata — NEVER
    hardcoded. Different bags have different layouts (the fixture bag is point_step=20
    with a field named `reflectivity`; the live Mid-360S is point_step=26 with
    `intensity`). Hardcoding either silently mis-reads the other.
  * Intensity is normalized by a FLAT divide (default /255.0), matching the inference
    side exactly (cuda_pp_ros.cpp: `reflectivity / REFLECTIVITY_SCALE`, no repair).
    Training must match inference. Corrupt input is NOT laundered — it is reported loudly.
  * Everything is streamed message-by-message; the whole bag is never held in RAM.
  * Deterministic: same bag + same args => byte-identical output.

Dependencies: Python 3.9+, numpy, rosbag2_py, sensor_msgs_py (ROS2 Humble).
Run with ROS sourced and conda deactivated:
    conda deactivate; source /opt/ros/humble/setup.bash
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


PC2_TYPE = "sensor_msgs/msg/PointCloud2"

# PointField.datatype enum -> human name (sensor_msgs/msg/PointField)
DATATYPE_NAME = {
    1: "INT8", 2: "UINT8", 3: "INT16", 4: "UINT16",
    5: "INT32", 6: "UINT32", 7: "FLOAT32", 8: "FLOAT64",
}
FLOAT32 = 7

# Intensity field name preference (matches cuda_pp_ros.cpp: intensity first, then reflectivity)
INTENSITY_NAMES = ("intensity", "reflectivity")


def eprint(*args, **kwargs) -> None:
    """Print to stderr so warnings are never swallowed by stdout redirection."""
    print(*args, file=sys.stderr, **kwargs)


def warn(msg: str) -> None:
    """Emit a prominent, un-missable warning."""
    bar = "!" * 78
    eprint("\n" + bar)
    for line in msg.splitlines():
        eprint("!! " + line)
    eprint(bar + "\n")


def git_commit(path: str) -> Optional[str]:
    """Best-effort git commit of the tree containing `path`. None if not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(path)), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def open_reader(bag_uri: str) -> SequentialReader:
    """Open a rosbag2 sqlite3 reader on `bag_uri` (the directory holding the .db3)."""
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def list_pc2_topics(bag_uri: str) -> list[tuple[str, str]]:
    """Return [(topic_name, type)] for every PointCloud2 topic in the bag."""
    reader = open_reader(bag_uri)
    topics = reader.get_all_topics_and_types()
    return [(t.name, t.type) for t in topics if t.type == PC2_TYPE]


def resolve_topic(bag_uri: str, requested: Optional[str]) -> str:
    """
    Resolve which topic to convert.

    - If `requested` is given: verify it exists and is PointCloud2, else exit.
    - Else autodetect: if exactly one PointCloud2 topic exists, use it (and say so).
      If zero or several exist, list them and exit — never silently guess.
    """
    pc2_topics = list_pc2_topics(bag_uri)
    names = [n for n, _ in pc2_topics]

    if requested is not None:
        if requested not in names:
            eprint(f"ERROR: requested --topic '{requested}' is not a "
                   f"{PC2_TYPE} topic in this bag.")
            eprint(f"       PointCloud2 topics present: {names or '(none)'}")
            sys.exit(2)
        print(f"[topic] using requested topic: {requested}")
        return requested

    if len(pc2_topics) == 1:
        chosen = pc2_topics[0][0]
        print(f"[topic] autodetected the single PointCloud2 topic: {chosen}")
        return chosen

    if len(pc2_topics) == 0:
        eprint(f"ERROR: no {PC2_TYPE} topics in this bag. Nothing to convert.")
        sys.exit(2)

    eprint("ERROR: multiple PointCloud2 topics found — pass --topic to choose one:")
    for n, _ in pc2_topics:
        eprint(f"         {n}")
    sys.exit(2)


def detect_layout(msg: PointCloud2) -> dict:
    """
    Inspect a message's fields and resolve x/y/z + intensity + (optional) tag by NAME.
    Fails loudly if x/y/z or an intensity-like field are missing.
    Returns a dict describing the layout, for logging and the manifest.
    """
    fields = {f.name: f for f in msg.fields}
    for req in ("x", "y", "z"):
        if req not in fields:
            eprint(f"ERROR: message is missing required field '{req}'. "
                   f"Fields present: {list(fields)}")
            sys.exit(3)

    intensity_name = next((n for n in INTENSITY_NAMES if n in fields), None)
    if intensity_name is None:
        eprint(f"ERROR: no intensity-like field found (looked for {INTENSITY_NAMES}). "
               f"Fields present: {list(fields)}")
        sys.exit(3)

    intensity_dt = fields[intensity_name].datatype
    if intensity_dt != FLOAT32:
        warn(
            f"Intensity field '{intensity_name}' has datatype "
            f"{DATATYPE_NAME.get(intensity_dt, intensity_dt)}, NOT FLOAT32.\n"
            f"It will be read per its actual datatype (values preserved), but the\n"
            f"live Mid-360S emits FLOAT32 — verify this bag is what you think it is."
        )

    tag_name = "tag" if "tag" in fields else None

    layout = {
        "point_step": int(msg.point_step),
        "frame_id": msg.header.frame_id,
        "is_dense": bool(msg.is_dense),
        "intensity_field": intensity_name,
        "intensity_datatype": DATATYPE_NAME.get(intensity_dt, intensity_dt),
        "tag_field": tag_name,
        "fields": [
            {
                "name": f.name,
                "offset": int(f.offset),
                "datatype": DATATYPE_NAME.get(f.datatype, f.datatype),
                "count": int(f.count),
            }
            for f in msg.fields
        ],
    }
    return layout


def frame_arrays(msg: PointCloud2, layout: dict):
    """
    Extract (xyz float64[N,3], intensity float64[N], tag int64[N] or None) from a
    message using sensor_msgs_py (which reads offsets/datatypes from metadata — no
    hardcoding). Returns raw values; filtering/normalization happen in the caller.
    """
    iname = layout["intensity_field"]
    want = ["x", "y", "z", iname]
    tname = layout["tag_field"]
    if tname:
        want.append(tname)

    # structured read handles mixed dtypes (float32 xyz/intensity + int tag)
    rec = pc2.read_points(msg, field_names=want, skip_nans=False)

    xyz = np.stack(
        [np.asarray(rec["x"], dtype=np.float64),
         np.asarray(rec["y"], dtype=np.float64),
         np.asarray(rec["z"], dtype=np.float64)],
        axis=1,
    )
    intensity = np.asarray(rec[iname], dtype=np.float64)
    tag = np.asarray(rec[tname]).astype(np.int64) if tname else None
    return xyz, intensity, tag


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a ROS2 rosbag of PointCloud2 into KITTI-format .bin frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--bag", required=True,
                    help="Path to the rosbag2 directory (the folder containing the .db3).")
    ap.add_argument("--topic", default=None,
                    help="PointCloud2 topic to convert. If omitted, autodetected only when "
                         "exactly one PointCloud2 topic exists; otherwise the script lists "
                         "them and exits (never silently guesses).")
    ap.add_argument("--out", required=True,
                    help="Output directory for the .bin files and manifest.json.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Maximum number of frames to write (default: all selected).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth message on the topic (e.g. 50 diverse frames from many).")
    ap.add_argument("--intensity-scale", type=float, default=255.0,
                    help="Divisor mapping intensity into the KITTI [0,1] domain. "
                         "Flat divide, matching cuda_pp_ros.cpp. No corruption repair.")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Escape hatch: write intensity as-is, skip the divide.")
    ap.add_argument("--drop-zeros", action=argparse.BooleanOptionalAction, default=True,
                    help="Drop (0,0,0) no-return points (default on). Use --no-drop-zeros to keep.")
    ap.add_argument("--tag-filter", action="store_true",
                    help="Keep only points with tag==0 (high confidence). Default off; the "
                         "tag distribution is reported regardless.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Inspect and report; write no files.")
    args = ap.parse_args()

    if args.stride < 1:
        eprint("ERROR: --stride must be >= 1.")
        return 2

    bag_uri = os.path.abspath(args.bag.rstrip("/"))
    if not os.path.isdir(bag_uri):
        eprint(f"ERROR: --bag is not a directory: {bag_uri}")
        return 2

    topic = resolve_topic(bag_uri, args.topic)
    scale = 1.0 if args.no_normalize else args.intensity_scale
    if not args.no_normalize and scale == 0.0:
        eprint("ERROR: --intensity-scale must be non-zero (or use --no-normalize).")
        return 2

    print(f"[cfg] bag={bag_uri}")
    print(f"[cfg] topic={topic}  stride={args.stride}  max_frames={args.max_frames}")
    print(f"[cfg] normalize={'OFF' if args.no_normalize else f'/{scale}'}  "
          f"drop_zeros={args.drop_zeros}  tag_filter={args.tag_filter}  dry_run={args.dry_run}")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    reader = open_reader(bag_uri)

    manifest = {
        "source_bag": bag_uri,
        "topic": topic,
        "intensity_scale": (None if args.no_normalize else scale),
        "normalize": (not args.no_normalize),
        "drop_zeros": bool(args.drop_zeros),
        "tag_filter": bool(args.tag_filter),
        "git_commit": git_commit(__file__),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detected_layout": None,
        "frames": [],
    }

    layout: Optional[dict] = None
    topic_msg_index = -1     # index among messages on the selected topic
    out_index = 0            # KITTI output index
    tag_hist_total: dict[int, int] = {}
    first_written = False

    while reader.has_next():
        rtopic, data, t = reader.read_next()
        if rtopic != topic:
            continue
        topic_msg_index += 1

        # frame selection: every `stride`-th message, up to max-frames
        if topic_msg_index % args.stride != 0:
            continue
        if args.max_frames is not None and out_index >= args.max_frames:
            break

        msg = deserialize_message(data, PointCloud2)
        if layout is None:
            layout = detect_layout(msg)
            manifest["detected_layout"] = layout
            print(f"[layout] point_step={layout['point_step']} frame_id={layout['frame_id']!r} "
                  f"intensity_field={layout['intensity_field']} "
                  f"({layout['intensity_datatype']}) tag_field={layout['tag_field']}")

        xyz, intensity, tag = frame_arrays(msg, layout)
        n_in = xyz.shape[0]

        # --- quality signal: (0,0,0) no-return rate (computed on raw input) ---
        zero_mask = (xyz == 0.0).all(axis=1)
        n_zeros = int(zero_mask.sum())
        zero_rate = (n_zeros / n_in) if n_in else 0.0
        nan_mask = ~np.isfinite(xyz).all(axis=1) | ~np.isfinite(intensity)
        n_nan = int(nan_mask.sum())

        # --- build keep mask ---
        keep = np.ones(n_in, dtype=bool)
        if args.drop_zeros:
            keep &= ~zero_mask
        keep &= ~nan_mask

        # tag distribution (always) + optional filter
        n_tag_dropped = 0
        if tag is not None:
            vals, counts = np.unique(tag, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                tag_hist_total[v] = tag_hist_total.get(v, 0) + c
            if args.tag_filter:
                tag_keep = (tag == 0)
                n_tag_dropped = int((keep & ~tag_keep).sum())
                keep &= tag_keep
        elif args.tag_filter:
            warn("--tag-filter requested but this bag has no 'tag' field; ignoring the filter.")

        xyz_k = xyz[keep]
        intensity_k = intensity[keep]
        n_out = xyz_k.shape[0]

        # --- intensity normalization (flat divide; NO repair) ---
        pre_min = float(intensity_k.min()) if n_out else float("nan")
        pre_max = float(intensity_k.max()) if n_out else float("nan")
        pre_mean = float(intensity_k.mean()) if n_out else float("nan")
        intensity_out = intensity_k if args.no_normalize else (intensity_k / scale)
        post_min = float(intensity_out.min()) if n_out else float("nan")
        post_max = float(intensity_out.max()) if n_out else float("nan")

        # --- loud detection on the FIRST written frame ---
        if not first_written:
            print(f"[intensity] first frame '{layout['intensity_field']}': "
                  f"pre  min/max/mean = {pre_min:.4f}/{pre_max:.4f}/{pre_mean:.4f}")
            print(f"[intensity] first frame after /{scale if not args.no_normalize else 1}: "
                  f"post min/max      = {post_min:.4f}/{post_max:.4f}")
            if not args.no_normalize:
                if pre_min < 0.0 or pre_max > 255.0:
                    warn(
                        f"Intensity outside [0,255] (pre min/max = {pre_min:.2f}/{pre_max:.2f}).\n"
                        f"A flat /{scale} will produce OUT-OF-DOMAIN values "
                        f"(post {post_min:.3f}/{post_max:.3f}).\n"
                        f"This is the signature of the legacy int8->uint8 corruption seen in the\n"
                        f"3-LiDAR fixture bag. It is NOT repaired (training must match inference,\n"
                        f"which also does a flat /255). Real Mid-360S data will not do this."
                    )
                elif pre_max <= 1.0:
                    warn(
                        f"Intensity already <= 1.0 (pre max = {pre_max:.4f}). Is this bag\n"
                        f"pre-normalized? A flat /{scale} would over-shrink it. Consider\n"
                        f"--no-normalize or --intensity-scale 1.0."
                    )
            first_written = True

        print(f"[frame {out_index:06d}] msg#{topic_msg_index} "
              f"pts {n_in}->{n_out}  zeros={n_zeros} ({zero_rate*100:.1f}%)  "
              f"nan={n_nan}  tag_dropped={n_tag_dropped}")

        # --- write (unless dry-run) ---
        if not args.dry_run:
            out_arr = np.empty((n_out, 4), dtype="<f4")
            out_arr[:, 0:3] = xyz_k.astype("<f4")
            out_arr[:, 3] = intensity_out.astype("<f4")
            out_path = os.path.join(args.out, f"{out_index:06d}.bin")
            out_arr.tofile(out_path)

        # header stamp (per-point time is a separate field we intentionally do not use here)
        stamp = msg.header.stamp
        manifest["frames"].append({
            "bin": f"{out_index:06d}.bin",
            "topic_msg_index": topic_msg_index,
            "bag_stamp_ns": int(t),
            "header_stamp_sec": int(stamp.sec),
            "header_stamp_nanosec": int(stamp.nanosec),
            "points_in": n_in,
            "points_out": n_out,
            "zeros_dropped": n_zeros,
            "zero_rate": round(zero_rate, 6),          # quality signal (mounting/aiming)
            "nan_dropped": n_nan,
            "tag_dropped": n_tag_dropped,
            "intensity_pre_min": pre_min,
            "intensity_pre_max": pre_max,
            "intensity_post_min": post_min,
            "intensity_post_max": post_max,
        })
        out_index += 1

    if layout is None:
        eprint(f"ERROR: no messages found on topic '{topic}'. Nothing converted.")
        return 4

    # --- summary ---
    frames = manifest["frames"]
    zero_rates = [f["zero_rate"] for f in frames]
    manifest["summary"] = {
        "frames_written": (0 if args.dry_run else len(frames)),
        "frames_selected": len(frames),
        "total_points_out": int(sum(f["points_out"] for f in frames)),
        "zero_rate_min": (min(zero_rates) if zero_rates else None),
        "zero_rate_max": (max(zero_rates) if zero_rates else None),
        "zero_rate_mean": (round(sum(zero_rates) / len(zero_rates), 6) if zero_rates else None),
        "tag_histogram_total": {str(k): int(v) for k, v in sorted(tag_hist_total.items())},
    }

    print(f"\n[summary] frames={len(frames)}  "
          f"zero_rate min/mean/max = "
          f"{manifest['summary']['zero_rate_min']}/"
          f"{manifest['summary']['zero_rate_mean']}/"
          f"{manifest['summary']['zero_rate_max']}")
    if tag_hist_total:
        print(f"[summary] tag histogram (all selected frames): "
              f"{manifest['summary']['tag_histogram_total']}")

    if args.dry_run:
        print("[dry-run] no .bin or manifest written.")
    else:
        manifest_path = os.path.join(args.out, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[done] wrote {len(frames)} .bin + {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
