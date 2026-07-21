#!/usr/bin/env python3
"""
mid360_dataset.py — OpenPCDet custom dataset for the Livox Mid-360S fine-tune set.

Subclasses `pcdet.datasets.DatasetTemplate`. The base class provides
prepare_data / data_augmentor / point_feature_encoder / data_processor /
collate_batch; this class only supplies data loading (`__init__`, `__len__`,
`__getitem__`), per docs/reports/AUDIT_dataset_template_*.

Contract fed to prepare_data (LiDAR frame, NO coordinate transform):
    input_dict = {
        frame_id : str
        points   : (N,4) float32  x,y,z,intensity(÷255)   # np.fromfile(bin).reshape(-1,4)
        gt_boxes : (M,7) float32  [x,y,z,dx,dy,dz,heading] # NO class id — base appends it
        gt_names : (M,)  str      'Car'/'Pedestrian'
    }
Frames come from data/derived/labels_openpcdet/MANIFEST_pairing.json; every path
is resolved under dataset_cfg.DATA_PATH so the same class runs on laptop and Colab
(no hardcoded absolute paths). All 48 frames are enumerated (overfit run).

Running this file directly executes the import-free smoke test (SMOKE TEST b),
which validates loading + shapes WITHOUT requiring a built `pcdet`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Guarded import: the class needs DatasetTemplate for real training, but the
# import-free smoke test (__main__) must run on a laptop with no pcdet build.
try:
    from pcdet.datasets import DatasetTemplate
    _HAS_PCDET = True
except Exception:
    DatasetTemplate = object  # fallback base so the class body still parses
    _HAS_PCDET = False

DEFAULT_MANIFEST_REL = "data/derived/labels_openpcdet/MANIFEST_pairing.json"


# --------------------------------------------------------------------------- #
# Loading helpers — used by BOTH __getitem__ and the import-free smoke test
# --------------------------------------------------------------------------- #
def read_points_bin(bin_path) -> np.ndarray:
    """KITTI-style read: (N,4) float32 x,y,z,intensity."""
    return np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)


def read_label_txt(label_path):
    """
    Parse `x y z dx dy dz heading category` lines.
    Returns (gt_boxes (M,7) float32, gt_names (M,) str). Empty file -> (0,7),(0,).
    """
    boxes, names = [], []
    if os.path.exists(str(label_path)):
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split()
                # 7 numeric + category (category may itself contain no spaces)
                boxes.append([float(v) for v in p[:7]])
                names.append(p[7])
    gt_boxes = (np.asarray(boxes, dtype=np.float32)
                if boxes else np.zeros((0, 7), dtype=np.float32))
    gt_names = np.asarray(names) if names else np.zeros((0,), dtype="<U16")
    return gt_boxes, gt_names


def load_frames(base_path, manifest_rel=DEFAULT_MANIFEST_REL):
    """Read the pairing manifest; return list of {frame_id, bin_path, label_path,
    num_boxes} with paths resolved under base_path."""
    base = Path(base_path)
    manifest_path = base / manifest_rel
    with open(manifest_path) as f:
        man = json.load(f)
    frames = []
    for e in man["frames"]:
        frames.append({
            "frame_id": e["frame_id"],
            "scene": e.get("scene"),
            "bin_path": base / e["intended_bin_path"],
            "label_path": base / e["txt_path"],
            "num_boxes": e.get("num_boxes"),
        })
    return frames


# --------------------------------------------------------------------------- #
# The dataset class
# --------------------------------------------------------------------------- #
class Mid360Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True,
                 root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger,
        )
        # self.root_path is set by the base to root_path or Path(DATA_PATH)
        manifest_rel = dataset_cfg.get("MANIFEST", DEFAULT_MANIFEST_REL) \
            if hasattr(dataset_cfg, "get") else DEFAULT_MANIFEST_REL
        self.frames = load_frames(self.root_path, manifest_rel)
        if self.logger is not None:
            self.logger.info(
                f"Mid360Dataset: {len(self.frames)} frames from "
                f"{self.root_path / manifest_rel}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        points = read_points_bin(frame["bin_path"])            # (N,4)
        gt_boxes, gt_names = read_label_txt(frame["label_path"])  # (M,7),(M,)

        input_dict = {
            "frame_id": frame["frame_id"],
            "points": points,
            "gt_boxes": gt_boxes,      # (M,7) — base prepare_data appends class id -> (M,8)
            "gt_names": gt_names,      # (M,) strings
        }
        return self.prepare_data(data_dict=input_dict)


# --------------------------------------------------------------------------- #
# SMOKE TEST b — import-free (no pcdet). Validates loading + shapes only.
# --------------------------------------------------------------------------- #
def _smoke_test_b():
    base = Path(__file__).resolve().parents[2]   # repo root: .../perception
    print(f"[smoke-b] repo root (DATA_PATH) = {base}")
    print(f"[smoke-b] pcdet importable here: {_HAS_PCDET} (False expected on laptop)")
    frames = load_frames(base)
    print(f"[smoke-b] frames in manifest: {len(frames)}")

    # 3 frames across different scenes
    pick = {}
    for fr in frames:
        pick.setdefault(fr["scene"], fr)
    sample = list(pick.values())[:3]

    allow = {"Car", "Pedestrian"}
    print(f"\n{'scene/frame':22s} {'N(pts)':>8s} {'pts.shape':>10s} "
          f"{'i∈[0,1]':>8s} {'M':>4s} {'gt_boxes':>10s} {'names ok':>9s} "
          f"{'M==lines':>9s} {'N==bytes/16':>12s} {'M==manifest':>12s}")
    ok_all = True
    for fr in sample:
        pts = read_points_bin(fr["bin_path"])
        gt_boxes, gt_names = read_label_txt(fr["label_path"])
        n_bytes = os.path.getsize(fr["bin_path"])
        n_lines = sum(1 for ln in open(fr["label_path"]) if ln.strip())

        a_shape = (pts.ndim == 2 and pts.shape[1] == 4 and pts.shape[0] > 0
                   and pts.dtype == np.float32)
        a_inten = (pts[:, 3].min() >= 0.0) and (pts[:, 3].max() <= 1.0)
        a_boxshape = (gt_boxes.ndim == 2 and gt_boxes.shape[1] == 7
                      and gt_boxes.dtype == np.float32)
        a_names = (gt_names.shape[0] == gt_boxes.shape[0]
                   and set(gt_names.tolist()) <= allow)
        a_mlines = (gt_boxes.shape[0] == n_lines)
        a_nbytes = (pts.shape[0] == n_bytes // 16)
        a_mman = (gt_boxes.shape[0] == fr["num_boxes"])
        row_ok = all([a_shape, a_inten, a_boxshape, a_names, a_mlines,
                      a_nbytes, a_mman])
        ok_all &= row_ok
        print(f"{fr['scene']+'/'+fr['frame_id']:22s} {pts.shape[0]:>8d} "
              f"{str(pts.shape):>10s} {str(a_inten):>8s} {gt_boxes.shape[0]:>4d} "
              f"{str(gt_boxes.shape):>10s} {str(a_names):>9s} {str(a_mlines):>9s} "
              f"{str(a_nbytes):>12s} {str(a_mman):>12s}")

    # explicit shape confirmation on frame 0
    p0 = read_points_bin(sample[0]["bin_path"])
    b0, n0 = read_label_txt(sample[0]["label_path"])
    print(f"\n[smoke-b] frame0 points dtype={p0.dtype} shape={p0.shape} "
          f"intensity[min,max]=[{p0[:,3].min():.3f},{p0[:,3].max():.3f}]")
    print(f"[smoke-b] frame0 gt_boxes shape={b0.shape} dtype={b0.dtype} "
          f"(MUST be (M,7), NOT (M,8) — class id appended later by prepare_data)")
    print(f"[smoke-b] frame0 gt_names[:3]={n0[:3].tolist()}")
    print(f"\n[smoke-b] RESULT: {'ALL ASSERTS PASS' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    _smoke_test_b()
