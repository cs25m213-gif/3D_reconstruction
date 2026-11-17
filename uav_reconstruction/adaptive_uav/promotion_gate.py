"""Fail-closed deployment checks; no ML imports or device initialization."""
import argparse
import json
from pathlib import Path

from .utility_contracts import sha256_file


def allowed(report, checkpoint, kind):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        return False
    digest = sha256_file(checkpoint)
    if kind == 'student':
        return (report.get('beat_teacher') is True and report.get('checkpoint_sha256') == digest
                and report.get('evaluation_protocol') == 'sparse_anchor_fit_v1'
                and report.get('validation_pixels',0)>0 and not report.get('diagnostic_labels',True))
    return (report.get('promote') is True and report.get('full_video_validated') is True
            and report.get('causal_online_evidence') is True and report.get('diagnostic_labels') is False
            and report.get('utility_checkpoint_sha256') == digest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--kind',choices=('student','selector'),required=True)
    a=p.parse_args()
    try:
        ok=allowed(json.loads(a.report.read_text()),a.checkpoint,a.kind)
    except (OSError,ValueError,TypeError):
        ok=False
    raise SystemExit(0 if ok else 1)


if __name__=='__main__':
    main()
