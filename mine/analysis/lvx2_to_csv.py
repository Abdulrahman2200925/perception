#!/usr/bin/env python3
"""
Livox .lvx2 → CSV converter  (pure Python + numpy, no ROS / Livox SDK)

Empirically verified against Livox Mid-360 sample files.

Binary layout (all little-endian):
  PUBLIC HEADER  24 bytes  : signature[16] + version[4] + magic_u32
  PRIVATE HEADER  5 bytes  : frame_duration_u32 + device_count_u8
  DEVICE INFO    63 bytes × device_count
  FRAME DATA BLOCKS (repeated):
    Frame header 24 bytes  : current_offset_u64 + next_offset_u64 + frame_index_u64
    Packages (repeated until next_offset):
      Package header 27 bytes:
        [0]     version / flags
        [1:5]   source IP (LiDAR IP, 4 bytes)
        [5]     device byte
        [6]     flag
        [7:15]  timestamp (uint64, nanoseconds)
        [15]    udp counter (increments per package)
        [16]    constant 0x44
        [17]    data_type  (0x01 = Cartesian High-precision for Mid-360)
        [18:20] data_length (uint16, = 1344 = 96 points × 14 bytes)
        [20:27] reserved / padding
      Point data  (data_length bytes):
        Each 14-byte point: int32 x_mm, int32 y_mm, int32 z_mm, uint8 refl, uint8 tag
"""

import argparse
import io
import os
import struct
import sys

import numpy as np


# ── Fixed layout constants ──────────────────────────────────────────────────
MAGIC            = 0xAC0EA767
PUB_HDR_SZ       = 24
PRIV_HDR_SZ      = 5
DEV_INFO_SZ      = 63   # per device; empirically confirmed
FRAME_HDR_SZ     = 24
PKG_HDR_SZ       = 27   # empirically confirmed
POINT_SZ         = 14   # int32 x,y,z + uint8 refl + uint8 tag
DATA_TYPE_CART_HP = 0x01

FIRST_FRAME_OFFSET = PUB_HDR_SZ + PRIV_HDR_SZ  # + dev_count*DEV_INFO_SZ (added at runtime)

POINT_DTYPE = np.dtype([
    ('x',            '<i4'),
    ('y',            '<i4'),
    ('z',            '<i4'),
    ('reflectivity', 'u1'),
    ('tag',          'u1'),
])

CSV_HEADER = 'x,y,z,reflectivity,tag,frame_index\n'


# ── Helpers ──────────────────────────────────────────────────────────────────

def hex_block(data: bytes, indent: str = '  ', width: int = 16) -> str:
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hexpart = ' '.join(f'{b:02x}' for b in chunk)
        ascpart = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'{indent}+{i:03d}  {hexpart:<{width*3-1}}  {ascpart}')
    return '\n'.join(lines)


def parse_pkg_header(raw27: bytes):
    """Return (data_type, data_length_bytes) from a 27-byte package header."""
    data_type   = raw27[17]
    data_length = struct.unpack_from('<H', raw27, 18)[0]
    return data_type, data_length


# ── Main parser ───────────────────────────────────────────────────────────────

def convert(args):
    in_path  = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + '.csv'
    file_sz  = os.path.getsize(in_path)

    print(f'\n{"="*62}')
    print(f'  LVX2 → CSV CONVERTER')
    print(f'  Input : {in_path}  ({file_sz/1e6:.1f} MB)')
    print(f'  Output: {out_path}')
    print(f'{"="*62}\n')

    with open(in_path, 'rb') as f:

        # ── 1. Public header ────────────────────────────────────────────────
        sig     = f.read(16)
        version = list(f.read(4))
        magic,  = struct.unpack('<I', f.read(4))

        print(f'Signature : {sig}')
        print(f'Version   : {version}')
        print(f'Magic     : 0x{magic:08X}', end='')
        if magic != MAGIC:
            print(f'  ← ERROR: expected 0x{MAGIC:08X}  — aborting')
            sys.exit(1)
        print('  ✓')

        # ── 2. Private header ───────────────────────────────────────────────
        frame_dur, = struct.unpack('<I', f.read(4))
        dev_count  = f.read(1)[0]
        print(f'Frame dur : {frame_dur} ms')
        print(f'Devices   : {dev_count}')

        # ── 3. Device info (skip) ───────────────────────────────────────────
        first_frame_off = PUB_HDR_SZ + PRIV_HDR_SZ + dev_count * DEV_INFO_SZ
        f.seek(first_frame_off)

        # ── 4. Probe first frame / first package header (diagnostic) ────────
        f.seek(first_frame_off)
        raw_fhdr = f.read(FRAME_HDR_SZ)
        cur0, nxt0, fidx0 = struct.unpack('<QQQ', raw_fhdr)
        raw_pkg0 = f.read(40)

        print(f'\n--- Frame 0 header (raw 24 bytes) ---')
        print(hex_block(raw_fhdr))
        print(f'  current_offset = {cur0}  next_offset = {nxt0}  frame_index = {fidx0}')

        print(f'\n--- First package header + 13 bytes (raw 40 bytes at +{FRAME_HDR_SZ}) ---')
        print(hex_block(raw_pkg0))
        dt0, dl0 = parse_pkg_header(raw_pkg0[:PKG_HDR_SZ])
        print(f'  data_type[17]  = 0x{dt0:02x} ({dt0})'
              f'{"  ← Cartesian High-precision ✓" if dt0==DATA_TYPE_CART_HP else "  ← UNEXPECTED"}')
        print(f'  data_length    = {dl0} bytes  → {dl0//POINT_SZ} points per package')
        print(f'  Frame 0 payload = {nxt0 - cur0 - FRAME_HDR_SZ} bytes'
              f'  → {(nxt0 - cur0 - FRAME_HDR_SZ)//(PKG_HDR_SZ+dl0)} packages')

        # Sanity gate: abort if layout doesn't look right
        if cur0 != first_frame_off:
            print(f'\nERROR: cur_offset={cur0} ≠ expected {first_frame_off}. '
                  f'Device info size guess wrong — try 59 or 65.')
            sys.exit(1)
        if dl0 == 0 or dl0 % POINT_SZ != 0:
            print(f'\nERROR: data_length={dl0} not a multiple of {POINT_SZ}.'
                  f' Package header offset may be wrong.')
            sys.exit(1)

        # ── 5. Walk all frames ───────────────────────────────────────────────
        stats = dict(
            frames=0, pkgs=0, pts_total=0, pts_valid=0, pts_invalid=0,
            x_min=2**31, x_max=-(2**31),
            y_min=2**31, y_max=-(2**31),
            z_min=2**31, z_max=-(2**31),
            refl_sum=0.0, refl_min=255, refl_max=0,
            skipped_pkgs=0,
        )
        per_frame_pts = []  # track pts per frame for avg/spread

        print(f'\n--- Parsing (max_frames={args.max_frames or "all"}) ---')

        out_buf = io.BytesIO()    # write buffer, flushed every N frames
        BUF_FRAMES = 20

        with open(out_path, 'w', buffering=1 << 20) as csvf:
            csvf.write(CSV_HEADER)

            frame_off = first_frame_off

            while True:
                # ── frame boundary checks ───────────────────────────────────
                if frame_off + FRAME_HDR_SZ > file_sz:
                    break
                if args.max_frames and stats['frames'] >= args.max_frames:
                    break

                f.seek(frame_off)
                raw_fh = f.read(FRAME_HDR_SZ)
                if len(raw_fh) < FRAME_HDR_SZ:
                    break

                cur_off, nxt_off, fidx = struct.unpack('<QQQ', raw_fh)

                if nxt_off == 0 or nxt_off <= cur_off or nxt_off > file_sz:
                    print(f'  EOF / corrupt guard hit at frame {stats["frames"]} '
                          f'(nxt_off={nxt_off})')
                    break

                payload_sz = nxt_off - cur_off - FRAME_HDR_SZ
                payload    = f.read(payload_sz)

                frame_pts = 0
                pos = 0

                # ── walk packages within frame ──────────────────────────────
                while pos + PKG_HDR_SZ <= payload_sz:
                    pkg_hdr = payload[pos:pos + PKG_HDR_SZ]
                    if len(pkg_hdr) < PKG_HDR_SZ:
                        break

                    data_type, data_len = parse_pkg_header(pkg_hdr)
                    pkg_end = pos + PKG_HDR_SZ + data_len

                    if pkg_end > payload_sz:
                        break  # truncated package

                    if data_type != DATA_TYPE_CART_HP or data_len == 0:
                        stats['skipped_pkgs'] += 1
                        pos = pkg_end
                        continue

                    pt_raw = payload[pos + PKG_HDR_SZ:pkg_end]
                    n_complete = len(pt_raw) // POINT_SZ
                    if n_complete == 0:
                        pos = pkg_end
                        continue

                    pts = np.frombuffer(pt_raw[:n_complete * POINT_SZ], dtype=POINT_DTYPE)
                    stats['pts_total'] += len(pts)
                    stats['pkgs'] += 1

                    # filter invalid (no-return) points
                    valid_mask = ~((pts['x'] == 0) & (pts['y'] == 0) & (pts['z'] == 0))
                    invalid_n  = int((~valid_mask).sum())
                    stats['pts_invalid'] += invalid_n

                    if not args.include_invalid:
                        pts = pts[valid_mask]

                    if len(pts) == 0:
                        pos = pkg_end
                        continue

                    frame_pts += len(pts)

                    # update running stats (on valid points)
                    vp = pts if args.include_invalid else pts
                    stats['x_min'] = min(stats['x_min'], int(vp['x'].min()))
                    stats['x_max'] = max(stats['x_max'], int(vp['x'].max()))
                    stats['y_min'] = min(stats['y_min'], int(vp['y'].min()))
                    stats['y_max'] = max(stats['y_max'], int(vp['y'].max()))
                    stats['z_min'] = min(stats['z_min'], int(vp['z'].min()))
                    stats['z_max'] = max(stats['z_max'], int(vp['z'].max()))
                    stats['refl_min'] = min(stats['refl_min'], int(vp['reflectivity'].min()))
                    stats['refl_max'] = max(stats['refl_max'], int(vp['reflectivity'].max()))
                    stats['refl_sum'] += float(vp['reflectivity'].astype(np.float64).sum())
                    stats['pts_valid'] += len(pts)

                    # build output rows as a single uint8 buffer for speed
                    fi_col = np.full(len(pts), fidx, dtype='<i8')
                    rows = np.column_stack([
                        pts['x'].astype('<i4'),
                        pts['y'].astype('<i4'),
                        pts['z'].astype('<i4'),
                        pts['reflectivity'].astype('<i4'),
                        pts['tag'].astype('<i4'),
                        fi_col,
                    ])
                    # np.savetxt on an already-open text file
                    np.savetxt(csvf, rows, fmt='%d', delimiter=',', newline='\n')

                    pos = pkg_end

                stats['frames'] += 1
                per_frame_pts.append(frame_pts)

                if stats['frames'] % args.progress_every == 0:
                    pct = frame_off / file_sz * 100
                    print(f'  frame {stats["frames"]:5d}  '
                          f'frame_idx={fidx}  '
                          f'pts_this_frame={frame_pts:6,}  '
                          f'total_valid={stats["pts_valid"]:10,}  '
                          f'({pct:.1f}% of file)')

                frame_off = nxt_off

    # ── 6. Sanity / evaluation report ────────────────────────────────────────
    n = stats['pts_valid']
    print(f'\n{"="*62}')
    print(f'  SANITY / EVALUATION REPORT')
    print(f'{"="*62}')
    print(f'  Magic code      : 0x{magic:08X} ✓')
    print(f'  Version         : {version}')
    print(f'  Frame duration  : {frame_dur} ms')
    print(f'  Device count    : {dev_count}')
    print()
    print(f'  Frames parsed   : {stats["frames"]:,}')
    print(f'  Packages        : {stats["pkgs"]:,}')
    print(f'  Skipped pkgs    : {stats["skipped_pkgs"]:,}  (non-type-1)')
    print(f'  Total pts (raw) : {stats["pts_total"]:,}')
    print(f'  Invalid (0,0,0) : {stats["pts_invalid"]:,}  '
          f'({100*stats["pts_invalid"]/max(stats["pts_total"],1):.1f}%)')
    print(f'  Valid pts written: {n:,}')

    if stats['frames'] > 0:
        avg = sum(per_frame_pts) / len(per_frame_pts)
        print(f'  Avg valid pts/frame: {avg:,.0f}  '
              f'(Mid-360 ~10k expected at 10Hz / 100ms frame)')

    if n > 0:
        print()
        print(f'  X range: {stats["x_min"]/1000:.3f} m … {stats["x_max"]/1000:.3f} m')
        print(f'  Y range: {stats["y_min"]/1000:.3f} m … {stats["y_max"]/1000:.3f} m')
        print(f'  Z range: {stats["z_min"]/1000:.3f} m … {stats["z_max"]/1000:.3f} m')
        print()
        mean_r = stats['refl_sum'] / n
        print(f'  Reflectivity min : {stats["refl_min"]}')
        print(f'  Reflectivity max : {stats["refl_max"]}')
        print(f'  Reflectivity mean: {mean_r:.2f}')
        if stats['refl_min'] < 0 or stats['refl_max'] > 255:
            print('  *** WARN: reflectivity out of 0-255 range — offset wrong! ***')
        else:
            print('  Reflectivity range 0-255: OK ✓')

    print()
    print(f'  Output CSV: {out_path}')
    out_sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f'  CSV size  : {out_sz/1e6:.1f} MB')
    print(f'{"="*62}\n')

    if n > 0:
        # Flag obviously wrong ranges
        max_r = max(abs(stats['x_min']), abs(stats['x_max']),
                    abs(stats['y_min']), abs(stats['y_max']),
                    abs(stats['z_min']), abs(stats['z_max']))
        if max_r > 200_000:      # > 200 m in mm
            print('*** WARN: coordinate magnitudes > 200 m — still in mm? '
                  'Check unit conversion.')
        elif max_r < 100:        # < 0.1 m in mm
            print('*** WARN: very small coordinate values — already in metres? '
                  'Check unit conversion.')
        else:
            print('Coordinate range looks plausible (values in mm → divide by 1000 for metres) ✓')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert Livox Mid-360 .lvx2 to CSV (pure Python + numpy)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('input',
                    help='.lvx2 input file')
    ap.add_argument('-o', '--output',
                    help='CSV output path (default: <input>.csv)')
    ap.add_argument('--max-frames', type=int, default=None,
                    help='Stop after this many frames (omit = all)')
    ap.add_argument('--include-invalid', action='store_true',
                    help='Write (0,0,0) no-return points to CSV')
    ap.add_argument('--progress-every', type=int, default=50,
                    help='Print progress every N frames')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f'ERROR: file not found: {args.input}')
        sys.exit(1)

    convert(args)


if __name__ == '__main__':
    main()
