#!/usr/bin/env python3
"""
patch_openpcdet.py — make OpenPCDet @846cf3e build on PyTorch 2.x (Colab).

Applies the AUDIT_pytorch_compat fixes to the *compiled C++/CUDA ops only*
(never the Python model code, so the exported ONNX graph is unaffected):

  (a) THC removal      : drop `#include <THC/THC.h>` / `extern THCState *state;`
                         (THC headers removed in PyTorch 1.11) — pointnet2 ops.
  (b) AT_CHECK         : AT_CHECK(  ->  TORCH_CHECK(   (AT_CHECK removed in torch 2.x)
  (c) .data<T>()       : .data<T>(  ->  .data_ptr<T>(  (Tensor::data<T>() removed)

Idempotent: re-running is a no-op (the patterns no longer match once patched).
Only touches files under `<root>/pcdet/ops/` — asserts nothing under pcdet/models
is edited.

Usage:
    python patch_openpcdet.py [/content/OpenPCDet]
"""
import os
import re
import sys

# The 10 pointnet2 sources carrying the THC include + `extern THCState *state;`
THC_FILES = [
    "pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp",
    "pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp",
    "pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp",
    "pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp",
    "pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp",
]

OPS_EXTS = (".cpp", ".cu", ".h", ".cuh")

# lines to drop for the THC fix (matched after .strip())
THC_LINE_RE = [
    re.compile(r"^#include\s*<THC/THC\.h>\s*$"),
    re.compile(r"^extern\s+THCState\s*\*\s*state\s*;\s*$"),
    re.compile(r"^THCState\s*\*\s*state\s*=.*;\s*$"),
]
DATA_RE = re.compile(r"\.data<([^>]+)>\(")   # .data<T>(  ->  .data_ptr<T>(


def is_ops_path(p):
    return (os.sep + os.path.join("pcdet", "ops") + os.sep) in (p + os.sep)


def patch_thc(root):
    changed_files = 0
    dropped = 0
    for rel in THC_FILES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [thc] MISSING (skip): {rel}")
            continue
        assert is_ops_path(path), f"refusing to edit non-ops path: {path}"
        with open(path) as f:
            lines = f.readlines()
        kept = []
        drop_here = 0
        for ln in lines:
            if any(rx.match(ln.strip()) for rx in THC_LINE_RE):
                drop_here += 1
                continue
            kept.append(ln)
        if drop_here:
            with open(path, "w") as f:
                f.writelines(kept)
            changed_files += 1
            dropped += drop_here
            print(f"  [thc] {rel}: dropped {drop_here} line(s)")
    return changed_files, dropped


def walk_ops(root):
    ops = os.path.join(root, "pcdet", "ops")
    for dp, _dn, fns in os.walk(ops):
        for fn in fns:
            if fn.endswith(OPS_EXTS):
                yield os.path.join(dp, fn)


def patch_at_check(root):
    changed_files = 0
    total = 0
    for path in walk_ops(root):
        assert "/pcdet/models/" not in path.replace(os.sep, "/"), path
        s = open(path).read()
        n = s.count("AT_CHECK(")
        if n:
            s = s.replace("AT_CHECK(", "TORCH_CHECK(")
            open(path, "w").write(s)
            changed_files += 1
            total += n
            print(f"  [at_check] {os.path.relpath(path, root)}: {n} replaced")
    return changed_files, total


def patch_data_ptr(root):
    changed_files = 0
    total = 0
    for path in walk_ops(root):
        assert "/pcdet/models/" not in path.replace(os.sep, "/"), path
        s = open(path).read()
        new, n = DATA_RE.subn(r".data_ptr<\1>(", s)
        if n:
            open(path, "w").write(new)
            changed_files += 1
            total += n
            print(f"  [data_ptr] {os.path.relpath(path, root)}: {n} replaced")
    return changed_files, total


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/content/OpenPCDet"
    root = os.path.abspath(root)
    assert os.path.isdir(os.path.join(root, "pcdet", "ops")), \
        f"not an OpenPCDet root (no pcdet/ops): {root}"
    print(f"Patching OpenPCDet for PyTorch 2.x at: {root}\n")

    print("(a) THC removal:")
    thc_files, thc_lines = patch_thc(root)
    print("\n(b) AT_CHECK -> TORCH_CHECK:")
    ac_files, ac_n = patch_at_check(root)
    print("\n(c) .data<T>() -> .data_ptr<T>():")
    dp_files, dp_n = patch_data_ptr(root)

    total_files = thc_files + ac_files + dp_files
    print("\n---- summary ----")
    print(f"  THC       : {thc_files} files, {thc_lines} lines dropped")
    print(f"  AT_CHECK  : {ac_files} files, {ac_n} replaced")
    print(f"  data_ptr  : {dp_files} files, {dp_n} replaced")
    if total_files == 0:
        print("PATCH: already patched — no-op.")
    else:
        print(f"PATCH COMPLETE: {total_files} file-edits applied "
              f"(some files may be counted under >1 fix).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
