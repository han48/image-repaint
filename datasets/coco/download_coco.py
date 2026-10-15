"""
Download COCO 2017 Train dataset with resume support.
Usage: python datasets/coco/download_coco.py
"""
import os
import zipfile
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
ZIP_PATH = RAW_DIR / "train2017.zip"
EXTRACT_DIR = RAW_DIR / "train2017"


def download_with_resume(url: str, dest: Path) -> None:
    """Download url to dest, resuming if dest already exists partially."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_size = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming download from {existing_size / 1e9:.2f} GB …")

    if HAS_REQUESTS:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code == 416:
                # Range not satisfiable – already complete
                print("File already fully downloaded (server says 416).")
                return
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) + existing_size
            mode = "ab" if existing_size > 0 else "wb"
            bar = tqdm(total=total, initial=existing_size, unit="B",
                       unit_scale=True, desc="train2017.zip") if HAS_TQDM else None
            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    if bar:
                        bar.update(len(chunk))
            if bar:
                bar.close()
    else:
        # Fallback: urllib (no resume support)
        def _progress(block_num, block_size, total_size):
            pct = min(100, block_num * block_size * 100 / total_size) if total_size > 0 else 0
            print(f"\rtrain2017.zip: {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, _progress)
        print()


def download_coco():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        imgs = list(EXTRACT_DIR.glob("*.jpg"))
        print(f"Already extracted: {len(imgs)} images found at {EXTRACT_DIR}")
        return

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        print(f"Downloading COCO 2017 Train from {COCO_TRAIN_URL} …")
        print(f"Saving to {ZIP_PATH}  (~19 GB, may take a while)")
    else:
        print(f"Partial zip found ({ZIP_PATH.stat().st_size / 1e9:.2f} GB). Attempting to resume …")

    try:
        download_with_resume(COCO_TRAIN_URL, ZIP_PATH)
    except Exception as e:
        # Keep partial file so next run can resume
        print(f"\nDownload interrupted: {e}")
        print(f"Partial zip kept at {ZIP_PATH} — re-run this script to resume.")
        raise

    print(f"\nExtracting {ZIP_PATH} to {RAW_DIR} …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print("Extraction complete.")

    imgs = list(EXTRACT_DIR.glob("*.jpg"))
    print(f"Summary: {len(imgs)} images extracted to {EXTRACT_DIR}")


if __name__ == "__main__":
    download_coco()
