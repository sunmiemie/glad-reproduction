"""Build a portable source/weight/video ZIP; Python dependencies install on target."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bootstrap import bootstrap


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path, action='append', default=[])
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise SystemExit('Output exists; use a new filename: %s' % target)
    source_manifest = bootstrap(ROOT, verify_only=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    files = {}
    for rel in filter(None, tracked):
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError('Unexpected tracked path: ' + rel)
        files[rel] = path
    for name in source_manifest['weights']:
        files['weights/' + name] = ROOT / 'weights' / name
    source = ROOT / 'external' / 'yolov5'
    for path in source.rglob('*'):
        if path.is_file() and not any(part in {'.git', '__pycache__', '.pytest_cache'} for part in path.parts):
            files[path.relative_to(ROOT).as_posix()] = path
    for video in args.video:
        video = video.resolve(strict=True)
        rel = 'videos/' + video.name
        if rel in files:
            raise ValueError('Duplicate video filename: ' + video.name)
        files[rel] = video
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    if dirty:
        raise SystemExit('Commit project changes before building a reproducible bundle.')
    manifest = {
        'repository': 'https://github.com/sunmiemie/glad-reproduction',
        'revision': revision,
        'target': 'Windows 11 / NVIDIA RTX 4060 / Python 3.11 x64 / PyTorch 2.8.0 cu126',
        'note': 'Code, upstream source, weights and selected videos included. Python and pip dependencies are NOT bundled.',
        'files': {rel: {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for rel, path in sorted(files.items())},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for rel, path in sorted(files.items()):
            archive.write(path, 'glad_reproduction/' + rel,
                          compress_type=zipfile.ZIP_STORED if path.suffix.lower() == '.mp4' else zipfile.ZIP_DEFLATED)
        archive.writestr('glad_reproduction/MIGRATION_MANIFEST.json',
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('ZIP CRC verification failed: ' + bad)
    checksum = sha256(target)
    target.with_suffix(target.suffix + '.sha256').write_text(checksum + '  ' + target.name + '\n', encoding='ascii')
    print(json.dumps({'output': str(target), 'bytes': target.stat().st_size,
                      'sha256': checksum, 'files': len(files), 'revision': revision}, indent=2))


if __name__ == '__main__':
    main()
