"""Convert online JSONL events to timing CSVs and a readable run report."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    events=[json.loads(line) for line in (root/'online_events.jsonl').read_text().splitlines() if line.strip()]
    frames=[row for row in events if row['event']=='frame']
    updates=[row for row in events if row['event']=='map_update']
    write_csv(root/'frame_timing.csv',frames,['source_index','source_timestamp_s','selected','reason',
        'decode_seconds','cheap_check_seconds','map_update_seconds','frame_processing_seconds'])
    write_csv(root/'map_update_timing.csv',updates,['update_index','source_timestamp_s','seen_keyframes',
        'accepted','reason','registered_frames','sparse_points','median_reprojection_error_px','seconds'])
    timing=json.loads((root/'online_timing.json').read_text()) if (root/'online_timing.json').exists() else {}
    metrics=json.loads((root/'reconstruction_metrics.json').read_text()) if (root/'reconstruction_metrics.json').exists() else {}
    lines=['# Online reconstruction timing report','',f'Output: `{root}`','',
        'The input was consumed sequentially; retrieval used past references only.',
        'Online sparse mapping and final batch dense meshing are timed separately.',
        'These are measured processing times, not source-video timestamps.','']
    invocation=timing.get('latest_invocation',{})
    for key in ['online_seconds','final_dense_seconds','total_seconds']:
        if key in invocation:
            lines.append(f'- {key}: {invocation[key]:.2f} seconds ({invocation[key]/60:.2f} minutes)')
    lines+=['',f'- Sampled frame records: {len(frames)}',
            f'- Accepted map updates: {sum(row["accepted"] for row in updates)}',
            f'- Rejected map updates: {sum(not row["accepted"] for row in updates)}']
    for key in ['selected_frames','registered_frames','dense_points','mesh_faces','median_reprojection_error_px']:
        if key in metrics:
            lines.append(f'- {key}: {metrics[key]}')
    if 'registration_fraction' in metrics:
        lines.append(f'- Registered fraction of selected frames: {100*metrics["registration_fraction"]:.2f}%')
    lines+=['','## Incremental map updates','',
            '| Update | Video time (s) | Accepted | Cameras | Points | Processing (s) |',
            '|---|---:|---|---:|---:|---:|']
    for row in updates:
        lines.append(f'| {row["update_index"]} | {row["source_timestamp_s"]:.2f} | {row["accepted"]} | '
                     f'{row.get("registered_frames","-")} | {row.get("sparse_points","-")} | {row["seconds"]:.2f} |')
    runtime=root/'runtime_metrics.json'
    if runtime.exists():
        stages=json.loads(runtime.read_text())['stages']
        lines+=['','## Final batch stage times','','| Stage | Seconds |','|---|---:|']
        for name,stage in stages.items():
            if 'provided_by' not in stage:
                lines.append(f'| {name} | {stage["seconds"]:.2f} |')
    lines+=['','## Limits','',
            'Not a real-time performance guarantee. Camera-registration rate is not surface completeness.',
            'Scale remains arbitrary and geometric accuracy has not been checked against ground truth.',
            'Rejected bootstrap attempts and solver warnings are retained in the logs.','']
    path=root/'TIME_REPORT.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reconstruction',type=Path)
    report(parser.parse_args().reconstruction.resolve())
