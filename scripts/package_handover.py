"""
Build a handover zip package.

Copies only the files that belong in the transfer, excluding PII-heavy
session recordings, caches, and venv directories. Optionally includes a
small curated set of sample sessions so the receiving team can verify
replay + analysis without hardware.

Usage:
    python scripts/package_handover.py                 # full package
    python scripts/package_handover.py --no-samples    # code+docs only
    python scripts/package_handover.py --out ../my.zip
    python scripts/package_handover.py --samples-list "cmj_20260423_151233_67909427,balance_eo_two_20260422_224542_67909427"

The zip is written to the parent directory by default (alongside the
project root) so `git status` stays clean.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


# ── Top-level items (from project root) to include ────────────────────
INCLUDE_ROOT = [
    # Code + entry points
    "src", "scripts", "main.py", "app.py", "config.py",
    # Dependency + env
    "requirements.txt", ".python-version", ".gitignore",
    # Docs (all handover-relevant)
    "docs",
    "HANDOVER.md", "README.md",
    # App resources (ChArUco board, mediapipe cache dir structure)
    "resources",
]

# ── Items under data/ to include (calibration artifacts only) ─────────
# Sessions subfolders are handled separately via --samples-list.
INCLUDE_DATA_CALIB_ITEMS = [
    # Pattern-match files directly under data/calibration/
    "encoder_verify_*.json",
    "intrinsics_*.npz",
    "extrinsics_*.npz",
    "poses3d_*.npz",
    "world_frame.npz",
]

# ── Exclusion patterns (applied during copy) ──────────────────────────
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc", "*.pyo",
    ".venv", "venv", "env",
    "*.bak",
    "_phase_*.py", "_scratch_*.py",
    ".DS_Store", "Thumbs.db",
]

# ── Default sample sessions (override with --samples-list) ────────────
# These are representative — a mix of tests so handover verifies cover
# + charts + replay across the pipeline.
DEFAULT_SAMPLES = [
    # Balance — different conditions
    "balance_eo_two_20260422_224542_67909427",
    "balance_ec_20260422_175433_67909427",
    # CMJ — visual flight phase
    "cmj_20260423_151233_67909427",
    # Squat — rep detection
    "squat_20260422_012329",
    # Free exercise — newest pipeline
    "free_exercise_20260423_182527_67909427",
]


def _ignore(src, names):
    """shutil.copytree ignore callback implementing EXCLUDE_PATTERNS."""
    out = set()
    from fnmatch import fnmatch
    for n in names:
        for pat in EXCLUDE_PATTERNS:
            if fnmatch(n, pat):
                out.add(n)
                break
    return out


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_samples(project: Path, staging: Path, sample_names: list[str]) -> int:
    """Copy a specific set of session folders. Returns count copied."""
    if not sample_names:
        return 0
    src_dir = project / "data" / "sessions"
    dst_dir = staging / "data" / "sessions"
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in sample_names:
        s = src_dir / name
        if not s.exists():
            print(f"  [miss] sample '{name}' not in data/sessions — skipping")
            continue
        d = dst_dir / name
        # Per-session exclusions: we *do* want mp4 + csv for replay, but
        # exclude .bak files.
        shutil.copytree(s, d, ignore=_ignore, dirs_exist_ok=True)
        count += 1
        print(f"  [ok]   sample {name}")
    return count


def _copy_calib(project: Path, staging: Path) -> None:
    """Copy calibration result files (not the raw session captures)."""
    from fnmatch import fnmatch
    src = project / "data" / "calibration"
    dst = staging / "data" / "calibration"
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for p in src.iterdir():
        if p.is_dir():
            # Skip session_* raw captures (large)
            continue
        for pat in INCLUDE_DATA_CALIB_ITEMS:
            if fnmatch(p.name, pat):
                shutil.copy2(p, dst / p.name)
                break


def _zip_staging(staging: Path, out_path: Path, root_folder: str) -> None:
    """Zip contents with a top-level folder so extracting creates
    ``<root_folder>/src/...`` rather than polluting the target directory."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                          compresslevel=6) as zf:
        for p in staging.rglob("*"):
            if p.is_file():
                rel = Path(root_folder) / p.relative_to(staging)
                zf.write(p, rel)


def _sha256(path: Path, buf: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default=None,
        help="output zip path (default: ../biomech-mocap-handover-<ts>.zip)")
    ap.add_argument("--no-samples", action="store_true",
        help="skip copying sample sessions (code+docs only)")
    ap.add_argument("--samples-list", type=str, default=None,
        help="comma-separated session folder names to include "
             "(default: curated DEFAULT_SAMPLES)")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (
        project.parent / f"biomech-mocap-handover-{ts}.zip")

    with tempfile.TemporaryDirectory(prefix="biomech_handover_") as tmp:
        staging = Path(tmp) / "biomech-mocap"
        staging.mkdir(parents=True, exist_ok=True)

        print(f"[package] project={project}")
        print(f"[package] staging={staging}")

        # ── Copy code + docs + top-level items ──────────────────────
        for name in INCLUDE_ROOT:
            src = project / name
            dst = staging / name
            print(f"  copy: {name}")
            _copy_tree(src, dst)

        # ── Copy calibration result files (not raw captures) ─────────
        print("  copy: data/calibration/ (result files only)")
        _copy_calib(project, staging)

        # ── Samples (optional) ──────────────────────────────────────
        if args.no_samples:
            print("  [skip] samples (--no-samples)")
        else:
            if args.samples_list:
                sample_names = [s.strip() for s in args.samples_list.split(",") if s.strip()]
            else:
                sample_names = DEFAULT_SAMPLES
            print(f"  copy: {len(sample_names)} sample session(s)")
            _copy_samples(project, staging, sample_names)

        # ── Zip ─────────────────────────────────────────────────────
        print(f"[package] compressing → {out_path} ...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_staging(staging, out_path, root_folder="biomech-mocap")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    checksum = _sha256(out_path)
    print()
    print(f"[done] {out_path}")
    print(f"       size   : {size_mb:.1f} MB")
    print(f"       sha256 : {checksum}")
    print()
    print("Next steps for the receiving team:")
    print("  1. Unzip to target path")
    print("  2. cd biomech-mocap && git init && git add -A && git commit -m 'Initial import'")
    print("  3. Create a venv with Python 3.11 and pip install -r requirements.txt")
    print("  4. Read HANDOVER.md § 10 (checklist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
