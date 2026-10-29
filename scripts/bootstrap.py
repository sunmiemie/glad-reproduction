"""Fetch pinned, hash-verified upstream YOLOv5 source and GLAD checkpoints.

Only uses public GitHub URLs. Upstream code/weights remain outside Git. Existing
source modifications are not overwritten; verification reports them instead.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("source_manifest.json")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, expected):
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 120))
            response.raise_for_status()
            data = response.content
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}")
            return data
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not fetch verified artifact from {url}: {error}")


def verify_yolo(target, manifest):
    marker = target / ".glad_revision"
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != manifest["revision"]:
        raise RuntimeError(f"Unrecognized source at {target}; expected pinned revision {manifest['revision']}")
    for name, expected in manifest["runtime_files"].items():
        path = target / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Upstream source missing or modified: {path}")


def bootstrap(root=ROOT, verify_only=False):
    root = Path(root).resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    weight_dir = root / "weights"
    if not verify_only:
        weight_dir.mkdir(parents=True, exist_ok=True)
    glad_revision = manifest["glad"]["revision"]
    for name, entry in manifest["weights"].items():
        path = weight_dir / name
        if path.is_file():
            if digest(path) != entry["sha256"]:
                raise RuntimeError(f"Existing checkpoint hash mismatch: {path}; preserve/move it before retrying")
        elif verify_only:
            raise FileNotFoundError(path)
        else:
            print(f"Downloading {name} ...", flush=True)
            url = f"https://raw.githubusercontent.com/WindyLab/Global-Local-MAV-Detection/{glad_revision}/weights/{name}"
            data = download(url, entry["sha256"])
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        print(f"Verified {name}", flush=True)

    target = root / "external" / "yolov5"
    yolo = manifest["yolov5"]
    if not target.exists():
        if verify_only:
            raise FileNotFoundError(target)
        print("Downloading pinned YOLOv5 v6.0 source ...", flush=True)
        data = download(f"https://codeload.github.com/ultralytics/yolov5/zip/{yolo['revision']}",
                        yolo["archive_sha256"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="glad-yolo-", dir=str(target.parent)) as folder:
            destination = Path(folder)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    name = PurePosixPath(member.filename)
                    if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                        raise ValueError("Unsafe path in upstream source archive")
                archive.extractall(destination)
            extracted = destination / ("yolov5-" + yolo["revision"])
            (extracted / ".glad_revision").write_text(yolo["revision"] + "\n", encoding="utf-8")
            verify_yolo(extracted, yolo)
            shutil.move(str(extracted), str(target))
    verify_yolo(target, yolo)
    print(f"Verified YOLOv5 v6.0 ({yolo['revision']})", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project destination (default: this repository)")
    parser.add_argument("--verify-only", action="store_true", help="check local files without network or writes")
    args = parser.parse_args()
    bootstrap(args.root, args.verify_only)


if __name__ == "__main__":
    main()
