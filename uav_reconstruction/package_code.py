"""Create a source-only reconstruction ZIP using an explicit allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'adaptive_uav_code_20261008.zip')
    args = parser.parse_args()
    sources = []
    for relative in ('adaptive_uav', 'configs', 'tests', 'drone_telemetry/scripts'):
        sources.extend(p for p in (root / relative).rglob('*')
                       if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md', '.html')
                       and '__pycache__' not in p.parts)
    sources.extend(root / relative for relative in (
        'package_code.py', 'requirements-live.txt', 'README_PORTABLE.md', 'WORK_DONE.md',
        'apps/openMVS-source/scripts/python/MvsUtils.py',
        'apps/openMVS-source/LICENSE', 'apps/openMVS-source/COPYRIGHT.md'))
    manifest = []
    # Exclusive creation protects an existing archive from accidental overwrite.
    with ZipFile(args.output, 'x', compression=ZIP_DEFLATED) as archive:
        for source in sorted(set(sources)):
            relative = str(source.relative_to(root))
            content = source.read_bytes()
            manifest.append({'path': relative, 'bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest()})
            archive.writestr('uav_reconstruction/' + relative, content)
        archive.writestr('uav_reconstruction/MANIFEST.json', json.dumps(manifest, indent=2))
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    print(f'Created {args.output.resolve()} ({args.output.stat().st_size:,} bytes; {len(sources)} source files)')


if __name__ == '__main__':
    main()
