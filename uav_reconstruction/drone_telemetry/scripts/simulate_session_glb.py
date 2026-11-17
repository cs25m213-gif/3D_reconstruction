#!/usr/bin/env python3
"""Export a raw-session GPS trajectory as a minimal, dependency-free GLB."""
import argparse, json, math, struct
from pathlib import Path

def read_gps(path):
    lines = path.read_text().splitlines()
    header = lines[0].split(',')
    out = []
    for line in lines[1:]:
        row = dict(zip(header, line.split(',')))
        try: out.append((float(row['latitude']), float(row['longitude']), float(row['altitude_m'])))
        except (KeyError, ValueError): pass
    return out

def make_glb(points, output):
    positions = b''.join(struct.pack('<3f', *p) for p in points)
    indices = b''.join(struct.pack('<I', i) for i in range(len(points)))
    blob = positions + indices
    while len(blob) % 4: blob += b'\0'
    doc = {
      'asset': {'version': '2.0', 'generator': 'CPU UAV trajectory simulator'},
      'scene': 0, 'scenes': [{'nodes': [0]}],
      'nodes': [{'mesh': 0, 'name': 'UAV GPS trajectory'}],
      'meshes': [{'name': 'GPS trajectory', 'primitives': [{
        'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 3
      }]}],
      'buffers': [{'byteLength': len(blob)}],
      'bufferViews': [
        {'buffer': 0, 'byteOffset': 0, 'byteLength': len(positions), 'target': 34962},
        {'buffer': 0, 'byteOffset': len(positions), 'byteLength': len(indices), 'target': 34963}],
      'accessors': [
        {'bufferView': 0, 'componentType': 5126, 'count': len(points), 'type': 'VEC3',
         'min': [min(p[i] for p in points) for i in range(3)],
         'max': [max(p[i] for p in points) for i in range(3)]},
        {'bufferView': 1, 'componentType': 5125, 'count': len(points), 'type': 'SCALAR'}]
    }
    js = json.dumps(doc, separators=(',', ':')).encode()
    while len(js) % 4: js += b' '
    glb = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(blob))
    glb += struct.pack('<I4s', len(js), b'JSON') + js
    glb += struct.pack('<I4s', len(blob), b'BIN\0') + blob
    output.write_bytes(glb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    gps = read_gps(args.session / 'gps.csv')
    if len(gps) < 2: raise SystemExit('Need at least two valid GPS records')
    lat0, lon0, alt0 = gps[0]
    # Local east/north approximation; altitude is relative to the first sample.
    radius = 6378137.0
    points = []
    for lat, lon, alt in gps:
        east = math.radians(lon - lon0) * radius * math.cos(math.radians(lat0))
        north = math.radians(lat - lat0) * radius
        points.append((east, north, alt - alt0))
    out = args.output or (args.session / 'uav_trajectory_simulation.glb')
    make_glb(points, out)
    print(f'Wrote {out} ({len(points)} GPS points)')

if __name__ == '__main__': main()
