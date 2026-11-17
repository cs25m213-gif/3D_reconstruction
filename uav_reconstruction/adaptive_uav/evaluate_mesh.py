"""Score a finished mesh against independent reference points, with failures left visible.

For complete videos. The reference and its ROI are fixed before the mesh is read. Scale comes from
outside the reference: pass the metres-per-unit you already established, because fitting it here
to the same points would make the accuracy number meaningless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .open3d_cpu import cpu_bindings
from .reconstruction_quality import ReferencePopulation, surface_metrics
from .utility_contracts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Point cloud file, or .npz with an array named points')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metres-per-unit', type=float, default=1.)
    parser.add_argument('--tolerance-m', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=2000000)
    parser.add_argument('--status', default='UNREVIEWED', help='Provenance of the reference, recorded with the result')
    args = parser.parse_args()
    o3d = cpu_bindings()
    if args.reference.suffix == '.npz':
        with np.load(args.reference) as data:
            target = data['points'].astype(np.float64)
    else:
        target = np.asarray(o3d.io.read_point_cloud(str(args.reference)).points)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not len(mesh.triangles):
        raise SystemExit('Mesh has no triangles')
    points = np.asarray(mesh.sample_points_uniformly(args.samples, seed=0).points) * args.metres_per_unit
    metrics = surface_metrics(points, ReferencePopulation(target, args.tolerance_m, args.status))
    report = {**metrics, 'mesh': str(args.mesh), 'reference': str(args.reference), 'reference_status': args.status, 'metres_per_unit': args.metres_per_unit, 'tolerance_m': args.tolerance_m}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
