"""Transfer RGB from an OpenMVS dense cloud to its mesh vertices."""

import argparse
import mmap
from pathlib import Path
import re
import struct

import numpy as np
from scipy.spatial import cKDTree


_DENSE_VERTEX = struct.Struct('<fffBBBfff')
_RGB_VERTEX = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def _header(data):
    end = data.find(b'end_header\n')
    if end < 0:
        raise ValueError('Missing PLY header terminator')
    end += len(b'end_header\n')
    header = data[:end].decode('ascii')
    if 'format binary_little_endian 1.0' not in header:
        raise ValueError('Only little-endian binary PLY is supported')
    counts = {name: int(count) for name, count in
              re.findall(r'^element (vertex|face) (\d+)$', header, re.M)}
    return header, end, counts


def _dense_points(path):
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header, offset, counts = _header(data)
        expected = ('property float32 x', 'property float32 y', 'property float32 z',
                    'property uint8 red', 'property uint8 green', 'property uint8 blue',
                    'property float32 nx', 'property float32 ny', 'property float32 nz',
                    'property list uint8 uint32 view_indices',
                    'property list uint8 float32 view_weights')
        if not all(item in header for item in expected):
            raise ValueError('Unsupported dense PLY vertex layout')
        count = counts['vertex']
        xyz = np.empty((count, 3), dtype=np.float32)
        rgb = np.empty((count, 3), dtype=np.uint8)
        for i in range(count):
            x, y, z, red, green, blue, _, _, _ = _DENSE_VERTEX.unpack_from(data, offset)
            xyz[i] = x, y, z
            rgb[i] = red, green, blue
            offset += _DENSE_VERTEX.size
            offset += 1 + 4 * data[offset]
            offset += 1 + 4 * data[offset]
        if offset != len(data):
            raise ValueError(f'Dense PLY length mismatch: parsed {offset}, file {len(data)}')
    return xyz, rgb


def color_mesh(dense_ply, mesh_ply, output_ply, workers=2):
    """Write a vertex-colored triangle PLY, using nearest dense-point colors."""
    xyz, rgb = _dense_points(dense_ply)
    if not np.isfinite(xyz).all():
        raise ValueError('Dense points contain non-finite coordinates')
    tree = cKDTree(xyz)
    del xyz
    with Path(mesh_ply).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header, start, counts = _header(data)
        vertex_count, face_count = counts['vertex'], counts['face']
        vertex_section = header.split('element vertex ', 1)[1].split('element face ', 1)[0]
        if 'property float32 x\nproperty float32 y\nproperty float32 z\n' not in vertex_section:
            raise ValueError('Unsupported mesh vertex layout')
        vertices = np.frombuffer(data, dtype='<f4', count=vertex_count * 3,
                                 offset=start).reshape(-1, 3).copy()
        if not np.isfinite(vertices).all():
            raise ValueError('Mesh vertices contain non-finite coordinates')
        face_start = start + vertex_count * 12
        if len(data) - face_start != face_count * 13:
            raise ValueError('Only triangle meshes with uint32 indices are supported')
        target = Path(output_ply)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as out:
            out.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {vertex_count}\n'
                       'property float x\nproperty float y\nproperty float z\n'
                       'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                       f'element face {face_count}\nproperty list uchar uint vertex_indices\n'
                       'end_header\n').encode('ascii'))
            for start_index in range(0, vertex_count, 100_000):
                chunk = vertices[start_index:start_index + 100_000]
                distance, nearest = tree.query(chunk, workers=workers)
                if not np.isfinite(distance).all():
                    raise ValueError('Nearest-color lookup failed')
                records = np.empty(len(chunk), dtype=_RGB_VERTEX)
                for axis, name in enumerate(('x', 'y', 'z')):
                    records[name] = chunk[:, axis]
                for axis, name in enumerate(('red', 'green', 'blue')):
                    records[name] = rgb[nearest, axis]
                out.write(records.tobytes())
                print(f'Colored {min(start_index + len(chunk), vertex_count)}/{vertex_count} vertices',
                      flush=True)
            for offset in range(face_start, len(data), 8 * 1024 * 1024):
                out.write(data[offset:min(offset + 8 * 1024 * 1024, len(data))])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dense_ply', type=Path)
    parser.add_argument('mesh_ply', type=Path)
    parser.add_argument('output_ply', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(color_mesh(args.dense_ply, args.mesh_ply, args.output_ply, args.workers))


if __name__ == '__main__':
    main()
