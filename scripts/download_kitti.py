"""
KITTI Raw Data Downloader
=========================
Downloads a KITTI raw sync sequence + calibration files into data/kitti/.

Usage:
  pixi run download-kitti                        # downloads default sequence
  pixi run download-kitti -- --sequence 0002     # specific sequence number
  pixi run download-kitti -- --date 2011_09_30 --sequence 0020
  pixi run download-kitti -- --list              # show all available sequences

Files downloaded (from KITTI's public S3 bucket — no login required):
  data/kitti/<date>/<date>_drive_<seq>_sync/
    image_02/data/*.png          left color camera frames
    velodyne_points/data/*.bin   Velodyne HDL-64 point clouds
  data/kitti/<date>/
    calib_cam_to_cam.txt         camera intrinsics + rectification
    calib_velo_to_cam.txt        LiDAR-to-camera extrinsics

Sequence sizes (approximate):
  0001  ~380 MB   city,    114 frames   ← default (small, good for testing)
  0002  ~650 MB   city,    177 frames
  0005  ~820 MB   city,    154 frames
  0009  ~1.1 GB   residential, 174 frames
  0015  ~1.4 GB   road,    297 frames
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
DATA_DIR = WORKSPACE_ROOT / "data" / "kitti"

# KITTI public S3 bucket — no auth required
_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# date → list of available sync sequences
SEQUENCES = {
    "2011_09_26": ["0001", "0002", "0005", "0009", "0011", "0013",
                   "0014", "0015", "0017", "0019", "0020", "0022",
                   "0023", "0027", "0028", "0029", "0032", "0035",
                   "0036", "0039", "0046", "0048", "0051", "0052",
                   "0056", "0057", "0059", "0060", "0061", "0064",
                   "0070", "0079", "0084", "0086", "0087", "0091",
                   "0093", "0095", "0096", "0101", "0104", "0106",
                   "0113", "0117", "0119"],
    "2011_09_28": ["0001", "0002"],
    "2011_09_29": ["0004"],
    "2011_09_30": ["0016", "0018", "0020", "0027", "0028", "0033", "0034"],
    "2011_10_03": ["0027", "0034", "0042", "0047"],
}

DEFAULT_DATE = "2011_09_26"
DEFAULT_SEQ  = "0001"


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Progress:
    """Simple progress bar for urllib downloads."""
    def __init__(self, filename: str):
        self._name = filename
        self._seen = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        self._seen += block_size
        if total_size > 0:
            pct = min(100, self._seen * 100 // total_size)
            mb  = self._seen / 1_048_576
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  {mb:.1f} MB", end="", flush=True)
        else:
            mb = self._seen / 1_048_576
            print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name}")
    try:
        urllib.request.urlretrieve(url, dest, _Progress(dest.name))
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR {e.code}: {url}")
        raise
    print()   # newline after progress bar


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} ...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()   # remove zip after extraction
    print("done")


def _sequence_url(date: str, seq: str) -> str:
    return f"{_S3}/{date}_drive_{seq}/{date}_drive_{seq}_sync.zip"


def _calib_url(date: str) -> str:
    return f"{_S3}/{date}_calib.zip"


def _sequence_exists(date: str, seq: str) -> bool:
    seq_dir = DATA_DIR / date / f"{date}_drive_{seq}_sync"
    img_dir = seq_dir / "image_02" / "data"
    lid_dir = seq_dir / "velodyne_points" / "data"
    if not (img_dir.is_dir() and lid_dir.is_dir()):
        return False
    n_img = len(list(img_dir.glob("*.png")))
    n_bin = len(list(lid_dir.glob("*.bin")))
    return n_img > 0 and n_bin > 0


def _calib_exists(date: str) -> bool:
    return (DATA_DIR / date / "calib_cam_to_cam.txt").exists() and \
           (DATA_DIR / date / "calib_velo_to_cam.txt").exists()


# ── Core ──────────────────────────────────────────────────────────────────────

def download_sequence(date: str, seq: str, force: bool = False) -> Path:
    """Download one KITTI raw sync sequence. Returns the sequence directory."""
    seq_dir = DATA_DIR / date / f"{date}_drive_{seq}_sync"

    print(f"\n{'─' * 55}")
    print(f"  Sequence : {date}_drive_{seq}_sync")
    print(f"  Target   : {seq_dir}")
    print(f"{'─' * 55}")

    # ── Calibration ───────────────────────────────────────────────────────────
    if not force and _calib_exists(date):
        print("  Calibration : already present, skipping.")
    else:
        print("  Calibration files:")
        tmp = DATA_DIR / f"_calib_{date}.zip"
        _download(_calib_url(date), tmp)
        _extract(tmp, DATA_DIR)   # extracts to DATA_DIR/<date>/calib_*.txt

    # ── Sequence data ─────────────────────────────────────────────────────────
    if not force and _sequence_exists(date, seq):
        n_img = len(list((seq_dir / "image_02" / "data").glob("*.png")))
        print(f"  Sequence data : already present ({n_img} frames), skipping.")
    else:
        print("  Sequence data:")
        tmp = DATA_DIR / f"_seq_{date}_{seq}.zip"
        _download(_sequence_url(date, seq), tmp)
        _extract(tmp, DATA_DIR / date)

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n  Validating ...")
    img_dir = seq_dir / "image_02" / "data"
    lid_dir = seq_dir / "velodyne_points" / "data"
    n_img = len(list(img_dir.glob("*.png")))
    n_bin = len(list(lid_dir.glob("*.bin")))

    ok = n_img > 0 and n_bin > 0 and n_img == n_bin
    status = "\033[92mOK\033[0m" if ok else "\033[91mFAILED\033[0m"
    print(f"  Images   : {n_img}  |  LiDAR scans : {n_bin}  [{status}]")

    if not ok:
        print("  Download may be incomplete. Re-run with --force to retry.")
        sys.exit(1)

    print(f"\n  \033[92mReady.\033[0m Set KITTI_SEQ to run the pipeline:")
    print(f"  export KITTI_SEQ={seq_dir}")
    print(f"  pixi run verify2")
    return seq_dir


def list_sequences() -> None:
    print("\nAvailable KITTI raw sequences:\n")
    for date, seqs in SEQUENCES.items():
        print(f"  {date}:  {', '.join(seqs)}")
    print(f"\nDefault: --date {DEFAULT_DATE} --sequence {DEFAULT_SEQ}")
    print(f"\nExample:")
    print(f"  pixi run download-kitti -- --date 2011_09_26 --sequence 0005")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download a KITTI raw sync sequence into data/kitti/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date",     default=DEFAULT_DATE,
                        help=f"Recording date (default: {DEFAULT_DATE})")
    parser.add_argument("--sequence", default=DEFAULT_SEQ,
                        help=f"4-digit sequence number (default: {DEFAULT_SEQ})")
    parser.add_argument("--force",    action="store_true",
                        help="Re-download even if files already exist")
    parser.add_argument("--list",     action="store_true",
                        help="List all available sequences and exit")
    args = parser.parse_args()

    if args.list:
        list_sequences()
        return

    date = args.date
    seq  = args.sequence.zfill(4)

    if date not in SEQUENCES:
        print(f"Unknown date '{date}'. Run --list to see options.")
        sys.exit(1)
    if seq not in SEQUENCES[date]:
        print(f"Sequence {seq} not found for {date}. Run --list to see options.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_sequence(date, seq, force=args.force)


if __name__ == "__main__":
    main()
