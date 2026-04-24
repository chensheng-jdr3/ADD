#!/usr/bin/env python3
"""Reorganize images according to annotations CSV into `my_dataset` structure.

Usage:
  python scripts/reorganize_from_annotations.py \
      --csv baldder_tissue_classification/annotations.csv \
      --src-root baldder_tissue_classification \
      --dst-root my_dataset \
      [--commit] [--overwrite]

By default runs in dry-run mode and only prints planned operations.
Use `--commit` to actually copy files. `--overwrite` allows replacing existing files.
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path


PATIENT_RE = re.compile(r'pt_?\d+', flags=re.IGNORECASE)


def find_source_file(src_root: Path, filename: str, tissue: str):
    # try direct tissue subfolder first
    candidates = [src_root / tissue / filename, src_root / filename]
    for p in candidates:
        if p.exists():
            return p
    # fallback: recursive search for filename under src_root
    for p in src_root.rglob(filename):
        if p.is_file():
            return p
    return None


def extract_patient_id(filename: str) -> str:
    m = PATIENT_RE.search(filename)
    if m:
        return m.group(0)
    # fallback: try pattern like _ptXXX or ptXXX in other positions
    return 'unknown_patient'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='baldder_tissue_classification/annotations.csv', help='path to annotations CSV')
    parser.add_argument('--src-root', default='baldder_tissue_classification', help='source images root')
    parser.add_argument('--dst-root', default='my_dataset_baldder', help='destination dataset root')
    parser.add_argument('--commit', action='store_true', default=True, help='actually copy files (default: dry-run)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing destination files')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')
    if not src_root.exists():
        raise SystemExit(f'Source root not found: {src_root}')

    planned = []
    missing = []

    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # handle header detection: if header contains 'HLY' assume header present
        if header and 'HLY' in [h.strip() for h in header]:
            pass
        else:
            # rewind
            f.seek(0)
            reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 4:
                continue
            filename = row[0].strip()
            imaging = row[1].strip()  # modality: WLI/NBI
            tissue = row[2].strip()   # tissue type
            sub = row[3].strip()      # sub_dataset: train/val/test

            src_file = find_source_file(src_root, filename, tissue)
            patient = extract_patient_id(filename)

            dst_dir = dst_root / sub / tissue / patient / imaging
            dst_file = dst_dir / filename

            if src_file is None:
                missing.append((filename, tissue, sub))
                if args.verbose:
                    print(f'MISSING: {filename} (expected under {src_root}/{tissue}/)')
                continue

            planned.append((src_file, dst_file))

    # summary
    print(f'Planned operations: {len(planned)} files, missing: {len(missing)}')
    if missing:
        print('Missing samples (first 20):')
        for t in missing[:20]:
            print('  ', t)

    if not args.commit:
        print('\nDry-run mode. To perform copying, re-run with --commit')
        return

    # perform copies
    copied = 0
    skipped = 0
    for src, dst in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f'SKIP exists: {dst}')
            continue
        shutil.copy2(src, dst)
        copied += 1
        if args.verbose:
            print(f'COPY {src} -> {dst}')

    print(f'Copy complete. copied={copied} skipped={skipped} missing={len(missing)}')


if __name__ == '__main__':
    main()
