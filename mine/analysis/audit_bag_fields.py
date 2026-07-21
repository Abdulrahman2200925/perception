#!/usr/bin/env python3
"""
audit_bag_fields.py — READ-ONLY audit of Livox Mid-360S rosbags.

Opens one or more rosbag2 (sqlite3) bags, deserializes /livox/lidar PointCloud2
messages, and reports:
  * PHASE 2: the exact PointField layout of the FIRST message (name/offset/datatype/count),
    point_step, row_step, width, height, is_dense, is_bigendian, frame_id, point count.
  * PHASE 3: per-bag statistics over ~N evenly-spaced sampled frames — point-count
    min/max/mean, intensity min/max/mean, x/y/z spatial extent, and NaN/inf counts.

Strictly read-only: it opens bags READ_ONLY, deserializes in memory, and prints. It never
publishes, never writes any bag, never writes a .bin. The only output is stdout.

Usage:
    conda deactivate; source /opt/ros/humble/setup.bash
    python3 audit_bag_fields.py <bag_dir> [<bag_dir> ...] [--topic /livox/lidar] [--samples 5]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

DATATYPE_NAME = {
    1: "INT8", 2: "UINT8", 3: "INT16", 4: "UINT16",
    5: "INT32", 6: "UINT32", 7: "FLOAT32", 8: "FLOAT64",
}
INTENSITY_NAMES = ("intensity", "reflectivity")


def open_reader(bag_dir: str) -> SequentialReader:
    r = SequentialReader()
    r.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return r


def count_topic_msgs(bag_dir: str, topic: str) -> int:
    """Light first pass: count messages on `topic` WITHOUT deserializing."""
    r = open_reader(bag_dir)
    n = 0
    while r.has_next():
        t, _data, _ts = r.read_next()
        if t == topic:
            n += 1
    return n


def read_nth_msg(bag_dir: str, topic: str, target_indices: set[int]):
    """Second pass: deserialize ONLY the messages on `topic` whose index is in
    target_indices. Yields (index, PointCloud2)."""
    r = open_reader(bag_dir)
    idx = -1
    remaining = set(target_indices)
    while r.has_next() and remaining:
        t, data, _ts = r.read_next()
        if t != topic:
            continue
        idx += 1
        if idx in remaining:
            remaining.discard(idx)
            yield idx, deserialize_message(data, PointCloud2)


def print_layout(msg: PointCloud2) -> None:
    print("  frame_id     :", repr(msg.header.frame_id))
    print("  point_step   :", msg.point_step)
    print("  row_step     :", msg.row_step)
    print("  width        :", msg.width)
    print("  height       :", msg.height)
    print("  is_dense     :", msg.is_dense)
    print("  is_bigendian :", msg.is_bigendian)
    n = (len(msg.data) // msg.point_step) if msg.point_step else 0
    print(f"  point count  : {msg.width * msg.height}  (len(data)/point_step = {n})")
    print(f"  {'name':<14}{'offset':<8}{'datatype':<18}{'count'}")
    for f in msg.fields:
        dt = f"{f.datatype} = {DATATYPE_NAME.get(f.datatype, '?')}"
        print(f"  {f.name:<14}{f.offset:<8}{dt:<18}{f.count}")


def intensity_field_name(msg: PointCloud2) -> Optional[str]:
    names = [f.name for f in msg.fields]
    return next((n for n in INTENSITY_NAMES if n in names), None)


def frame_stats(msg: PointCloud2, iname: str):
    """Return dict of per-frame stats from one message (read-only)."""
    rec = pc2.read_points(msg, field_names=["x", "y", "z", iname], skip_nans=False)
    x = np.asarray(rec["x"], dtype=np.float64)
    y = np.asarray(rec["y"], dtype=np.float64)
    z = np.asarray(rec["z"], dtype=np.float64)
    inten = np.asarray(rec[iname], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)
    finite = np.isfinite(xyz).all(axis=1)
    nan_inf = int((~finite).sum())
    # zero-return (exact origin) and near-range (< 2 m, min-usable) counts
    rng = np.sqrt((xyz * xyz).sum(axis=1))
    n_zero = int((xyz == 0.0).all(axis=1).sum())
    n_near = int(((rng > 0.0) & (rng < 2.0)).sum())   # closer than 2 m, excluding exact origin
    # stats on finite points for extent; intensity over all (report separately)
    xf = xyz[finite]
    return {
        "n": int(x.size),
        "nan_inf": nan_inf,
        "n_zero": n_zero,
        "n_near": n_near,
        "x_min": float(xf[:, 0].min()) if xf.size else float("nan"),
        "x_max": float(xf[:, 0].max()) if xf.size else float("nan"),
        "y_min": float(xf[:, 1].min()) if xf.size else float("nan"),
        "y_max": float(xf[:, 1].max()) if xf.size else float("nan"),
        "z_min": float(xf[:, 2].min()) if xf.size else float("nan"),
        "z_max": float(xf[:, 2].max()) if xf.size else float("nan"),
        "i_min": float(np.nanmin(inten)) if inten.size else float("nan"),
        "i_max": float(np.nanmax(inten)) if inten.size else float("nan"),
        "i_mean": float(np.nanmean(inten)) if inten.size else float("nan"),
    }


def audit_bag(bag_dir: str, topic: str, n_samples: int, show_layout: bool) -> None:
    name = os.path.basename(bag_dir.rstrip("/"))
    total = count_topic_msgs(bag_dir, topic)
    print(f"\n===== BAG: {name}  ({topic}: {total} msgs) =====")
    if total == 0:
        print("  (no messages on topic)")
        return

    # evenly spaced sample indices across the whole bag
    if n_samples >= total:
        targets = list(range(total))
    else:
        targets = [round(i * (total - 1) / (n_samples - 1)) for i in range(n_samples)] \
            if n_samples > 1 else [0]
    targets = sorted(set(targets))

    stats = []
    iname = None
    for idx, msg in read_nth_msg(bag_dir, topic, set(targets)):
        if iname is None:
            iname = intensity_field_name(msg)
            if show_layout:
                print(f"\n--- PHASE 2: field layout (first sampled msg, index {idx}) ---")
                print_layout(msg)
                print(f"  intensity-like field found: {iname!r}")
                print("--- end layout ---")
            if iname is None:
                print("  ERROR: no intensity/reflectivity field; cannot compute intensity stats.")
                return
        s = frame_stats(msg, iname)
        s["idx"] = idx
        stats.append(s)

    # per-bag aggregation
    ns = [s["n"] for s in stats]
    print(f"\n  sampled frame indices: {[s['idx'] for s in stats]}")
    print(f"  intensity field read : {iname!r}")
    print(f"  point count   min/max/mean : {min(ns)} / {max(ns)} / {sum(ns)/len(ns):.1f}")
    imin = min(s["i_min"] for s in stats)
    imax = max(s["i_max"] for s in stats)
    imean = sum(s["i_mean"] for s in stats) / len(stats)
    print(f"  intensity     min/max/mean : {imin:.3f} / {imax:.3f} / {imean:.3f}")
    print(f"  x extent      min/max      : {min(s['x_min'] for s in stats):.3f} / {max(s['x_max'] for s in stats):.3f}")
    print(f"  y extent      min/max      : {min(s['y_min'] for s in stats):.3f} / {max(s['y_max'] for s in stats):.3f}")
    print(f"  z extent      min/max      : {min(s['z_min'] for s in stats):.3f} / {max(s['z_max'] for s in stats):.3f}")
    print(f"  NaN/inf xyz   total/frame  : {sum(s['nan_inf'] for s in stats)} over {len(stats)} frames")
    print(f"  (0,0,0) zero  total/frame  : {sum(s['n_zero'] for s in stats)} over {len(stats)} frames")
    print(f"  range < 2 m   total/frame  : {sum(s['n_near'] for s in stats)} over {len(stats)} frames")
    print("  per-frame:")
    for s in stats:
        print(f"    idx {s['idx']:>5}: n={s['n']:>6}  i[{s['i_min']:.1f},{s['i_max']:.1f}] mean {s['i_mean']:.1f}  "
              f"x[{s['x_min']:.1f},{s['x_max']:.1f}] y[{s['y_min']:.1f},{s['y_max']:.1f}] z[{s['z_min']:.1f},{s['z_max']:.1f}]  "
              f"nan/inf={s['nan_inf']} zero={s['n_zero']} near2m={s['n_near']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only audit of Livox rosbag PointCloud2 layout + stats.")
    ap.add_argument("bags", nargs="+", help="rosbag2 directories (folder containing the .db3).")
    ap.add_argument("--topic", default="/livox/lidar")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    for i, b in enumerate(args.bags):
        b = os.path.abspath(b.rstrip("/"))
        if not os.path.isdir(b):
            print(f"SKIP (not a dir): {b}", file=sys.stderr)
            continue
        audit_bag(b, args.topic, args.samples, show_layout=(i == 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
