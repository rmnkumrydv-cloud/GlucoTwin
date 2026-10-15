"""Step 5: get CGMacros (PhysioNet, open access) into data/raw/cgmacros.

    python scripts/download_data.py                        # download from PhysioNet and unzip
    python scripts/download_data.py --source aws           # same file from PhysioNet's AWS mirror
    python scripts/download_data.py --source figshare      # same data from the Scientific Data figshare
    python scripts/download_data.py --from-zip PATH.zip    # use a zip you downloaded in a browser

Any of these zips works with --from-zip, including the "Download the ZIP file" bundle from
PhysioNet and the figshare zip, which contain the data zip nested inside another zip.

Works on Windows, macOS and Linux with plain Python (no wget/curl needed). Interrupted downloads
resume. Meal photos are skipped when unzipping: the model does not need them.

The data is licensed CC BY-NC-SA 4.0 (PhysioNet). Never commit it to GitHub (.gitignore already
excludes it). Dataset page: https://physionet.org/content/cgmacros/1.0.0/
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files/cgmacros/1.0.0/"
ZIP_NAME = "CGMacros_dateshifted365.zip"
SOURCES = {
    "physionet": BASE + ZIP_NAME,
    "aws": "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/" + ZIP_NAME,
    "figshare": "https://ndownloader.figshare.com/files/56754167",   # bundle with the zip nested inside
}
KEEP_SUFFIXES = {".csv", ".txt"}


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Resumable download with a progress line."""
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "GlucoTwin-downloader",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # server ignored the range request: start again
                    have = 0
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    done, t0 = have, time.time()
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({rate:.1f} MB/s)  ")
                            sys.stdout.flush()
            print()
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:                    # already complete
                return
            print(f"\n  HTTP {exc.code} on attempt {attempt}: {exc.reason}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"\n  network problem on attempt {attempt}: {exc}")
        time.sleep(min(5 * attempt, 30))
    raise SystemExit(f"Download failed after {retries} attempts. Download {url} in a browser and run:\n"
                     f"  python scripts/download_data.py --from-zip <path to the downloaded zip>")


def safe_extract(zip_path: Path, out: Path, depth: int = 0) -> int:
    """Extract CSV/TXT files, refusing paths that would escape ``out``. Zips nested inside the zip
    (PhysioNet's "Download the ZIP file" bundle and the figshare zip) are opened and extracted too."""
    out = out.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name:
                continue
            suffix = Path(name).suffix.lower()
            if suffix == ".zip" and depth < 2:
                print(f"  opening nested zip {Path(name).name} ...")
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    inner = Path(tmp) / "inner.zip"
                    with z.open(info) as src, open(inner, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    n += safe_extract(inner, out, depth + 1)
                continue
            if suffix not in KEEP_SUFFIXES:
                continue
            target = (out / name).resolve()
            if out not in target.parents:
                raise SystemExit(f"Refusing suspicious path in zip: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "cgmacros"))
    ap.add_argument("--from-zip", default=None, help="use an already downloaded CGMacros zip")
    ap.add_argument("--source", choices=sorted(SOURCES), default="physionet",
                    help="where to download from (default physionet)")
    ap.add_argument("--base-url", default=None, help=argparse.SUPPRESS)  # used by the tests
    ap.add_argument("--keep-zip", action="store_true", help="keep the zip after unzipping")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_zip:
        zip_path = Path(args.from_zip.strip('"'))
        if not zip_path.exists():
            raise SystemExit(f"{zip_path} not found. In File Explorer, right-click the zip, choose "
                             "'Copy as path' and paste it after --from-zip.")
    else:
        url = (args.base_url + ZIP_NAME) if args.base_url else SOURCES[args.source]
        zip_path = out / f"download_{args.source}.zip"
        print(f"Downloading {url}\n  -> {zip_path}")
        download(url, zip_path)

    try:
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip() if zip_path.stat().st_size < 50e6 else None
        if bad:
            raise zipfile.BadZipFile(bad)
    except zipfile.BadZipFile:
        raise SystemExit(f"{zip_path} is incomplete or corrupt (the download did not finish). "
                         "Delete it and download again.")

    print("Unzipping CSV files (meal photos skipped)...")
    n = safe_extract(zip_path, out)
    print(f"  {n} files extracted to {out}")
    if not args.from_zip and not args.keep_zip:
        zip_path.unlink()

    parts = sorted(out.rglob("CGMacros-0*.csv"))
    bio = list(out.rglob("bio.csv"))
    print(f"\nFound {len(parts)} participant files and {len(bio)} bio.csv in {out}")
    if len(parts) < 40 or not bio:
        raise SystemExit(
            "The participant data was not in that zip. bio.csv and the CGMacros-0XX folders are inside "
            "CGMacros_dateshifted365.zip.\n"
            "  - If you downloaded the figshare / 'Download the ZIP file' bundle, make sure you are using the "
            "updated scripts/download_data.py (it opens the zip inside the zip), or copy "
            "CGMacros_dateshifted365.zip out of the bundle and pass that file to --from-zip.\n"
            "  - If the zip is smaller than about 650 MB, the download did not finish: download it again.")
    print("Next step:  python -m glucotwin.prepare")


if __name__ == "__main__":
    main()
