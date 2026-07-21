# bag_to_bin

Convert a ROS2 rosbag of `sensor_msgs/PointCloud2` into **KITTI-format `.bin`** frames for
OpenPCDet training / labeling.

```
rosbag (Mid-360S PointCloud2)  ->  bag_to_bin.py  ->  KITTI .bin  ->  labeling  ->  OpenPCDet
```

Each selected message becomes one file: `float32[N,4] = (x, y, z, intensity)`, flat
little-endian, no header, named `000000.bin`, `000001.bin`, … Plus a `manifest.json`.

## Requirements

ROS2 Humble (`rosbag2_py`, `sensor_msgs_py`) + `numpy`. Run with ROS sourced and conda off:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
```

## Usage

```bash
python3 bag_to_bin.py --bag <bag_dir> --out <out_dir> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--bag PATH` | (required) | rosbag2 directory (the folder holding the `.db3`) |
| `--topic NAME` | autodetect | PointCloud2 topic. See "Topic resolution" below. |
| `--out DIR` | (required) | output directory for `.bin` + `manifest.json` |
| `--max-frames N` | all | cap the number of frames written |
| `--stride N` | 1 | take every Nth message (e.g. 50 diverse frames from 1373) |
| `--intensity-scale F` | 255.0 | flat divisor into KITTI's [0,1] domain |
| `--no-normalize` | off | write intensity as-is (skip the divide) |
| `--drop-zeros` / `--no-drop-zeros` | on | drop `(0,0,0)` no-return points |
| `--tag-filter` | off | keep only `tag == 0` (high-confidence) points |
| `--dry-run` | off | inspect and report; write nothing |

### Examples

```bash
# inspect only
python3 bag_to_bin.py --bag ../../data/raw/bags/garage_2026-07-18 --out /tmp/g --dry-run

# 50 diverse frames (every ~27th message) from a real garage bag
python3 bag_to_bin.py --bag ../../data/raw/bags/garage_2026-07-18 \
    --out ../../data/derived/bin/garage_2026-07-18 --stride 27 --max-frames 50
```

## Topic resolution (why `--topic` is not a silent default)

The default topic is **not** hardcoded to `/livox/lidar`, because the same `.bin` producer
must work on bags whose topic differs (the dev fixture uses `kitti/velo`). Behaviour:

- `--topic` given → verified against the bag; exits if it isn't a PointCloud2 topic.
- omitted + **exactly one** PointCloud2 topic → that topic is used and announced.
- omitted + **zero or several** → the script **lists them and exits**. It never guesses.

## Intensity: flat ÷255, no repair (this is deliberate)

KITTI intensity is `[0,1]`; the Mid-360S publishes `intensity` FLOAT32 `[0,255]`. The script
divides by `--intensity-scale` (default 255).

**It does a flat divide and does NOT repair corrupt input.** The inference node
(`upstream/CUDA-PointPillars-ROS2/src/cuda_pp_ros.cpp`) applies `reflectivity / 255.0f` with
no repair — **training must match inference**. If training "fixed" a corruption inference
does not, the model would see one distribution in training and another at runtime — worse
than uniformly-corrupt data.

So the script **detects and shouts** instead of patching:

- intensity `< 0` or `> 255` → loud warning (signature of the legacy int8→uint8 corruption
  seen in the 3-LiDAR fixture bag). Output values will be out of `[0,1]` **by design** —
  that is honest, not a bug.
- intensity already `≤ 1` → warning that the bag may be pre-normalized.

It prints the first frame's pre- and post-normalization min/max so a wrong scale is visible
immediately.

### What "correct" output looks like

| Input | Expected output | Warning? |
|---|---|---|
| clean Mid-360S (`intensity` 0–255) | intensity in `[0,1]` | none |
| corrupt fixture (`reflectivity` −128…127) | intensity in `≈[−0.5,+0.5]` | **yes, loud** |

A converter that produced clean `[0,1]` output from corrupt `[-128,127]` input would be
**lying** about the data.

## `(0,0,0)` no-return points

The Mid-360S emits `(0,0,0)` when nothing is detected in range — "no return", not a real
point at the sensor. Kept, they form a phantom blob at the origin. `--drop-zeros` (default
**on**) removes them; NaN/Inf are always dropped.

The **zero-return rate is a quality signal**, not just a filter stat: it is printed per frame
and recorded in `manifest.json` (`zero_rate`, plus min/mean/max in `summary`). The dev
fixture shows ~37.7%. If a real garage bag shows a wildly different rate, that can indicate a
mounting/aiming problem worth catching early.

## `tag` byte

`tag` encodes confidence (bit-pairs: dragging-noise / atmospheric / other); `0` = high
confidence. The distribution is always reported (per-run histogram in the manifest).
`--tag-filter` keeps only `tag == 0`. Default **off** — decide with the data first.

## manifest.json

Written next to the `.bin` files. Records source bag, topic, intensity scale, the **detected
field layout**, git commit (if the tree is a repo), and per frame: output name, original
`topic_msg_index`, bag + header timestamps, points in/out, zeros dropped + `zero_rate`, NaN
dropped, tag dropped, and intensity pre/post min/max. **This is the only way back from
`000042.bin` to the exact source message** — needed once frames are labeled.

## What this is NOT

- **No de-skew** (needs odometry that doesn't exist yet; record bags stationary).
- **No accumulation** (the 3-frame / 300 ms runtime design is a separate ROS2 node). One
  `.bin` per message here.
- **No labeling, no config changes.** Separate phases.
