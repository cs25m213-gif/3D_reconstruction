#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def load_transform(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    scale = float(data["scale_m_per_colmap_unit"])
    R = np.asarray(data["rotation_matrix"], dtype=float)
    t = np.asarray(data["translation_m"], dtype=float)

    return scale, R, t


def transform_points(xyz, scale, R, t):
    return scale * (R @ xyz.T).T + t


def transform_normals(normals, R):
    transformed = (R @ normals.T).T

    norm = np.linalg.norm(
        transformed,
        axis=1,
        keepdims=True
    )

    norm[norm == 0] = 1.0

    return transformed / norm


def transform_ply(src, dst, scale, R, t):
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise RuntimeError(
            "plyfile is required.\n"
            "Install with: pip install plyfile"
        )

    print("Loading PLY:", src)

    ply = PlyData.read(str(src))

    vertex = ply["vertex"].data

    xyz = np.column_stack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).astype(np.float64)

    transformed = transform_points(
        xyz,
        scale,
        R,
        t
    )

    vertex["x"] = transformed[:, 0]
    vertex["y"] = transformed[:, 1]
    vertex["z"] = transformed[:, 2]

    names = vertex.dtype.names

    if all(
        name in names
        for name in ("nx", "ny", "nz")
    ):
        normals = np.column_stack([
            vertex["nx"],
            vertex["ny"],
            vertex["nz"]
        ]).astype(np.float64)

        normals = transform_normals(
            normals,
            R
        )

        vertex["nx"] = normals[:, 0]
        vertex["ny"] = normals[:, 1]
        vertex["nz"] = normals[:, 2]

        print("Rotated vertex normals.")

    elements = []

    for element in ply.elements:
        if element.name == "vertex":
            elements.append(
                PlyElement.describe(
                    vertex,
                    "vertex"
                )
            )
        else:
            elements.append(element)

    out = PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order
    )

    out.write(str(dst))

    print("Saved PLY:", dst)
    print("Vertices :", len(vertex))

    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)

    print()
    print("ENU bounds:")
    print(
        f"East : {mins[0]:.2f} .. {maxs[0]:.2f} m"
    )
    print(
        f"North: {mins[1]:.2f} .. {maxs[1]:.2f} m"
    )
    print(
        f"Up   : {mins[2]:.2f} .. {maxs[2]:.2f} m"
    )


def transform_obj(src, dst, scale, R, t):

    print("Transforming OBJ:", src)

    vertex_count = 0
    normal_count = 0

    with open(
        src,
        "r",
        errors="replace"
    ) as fin, open(
        dst,
        "w"
    ) as fout:

        for line in fin:

            if line.startswith("v "):

                values = line.split()

                if len(values) >= 4:

                    xyz = np.array([
                        float(values[1]),
                        float(values[2]),
                        float(values[3])
                    ])

                    xyz = (
                        scale * (R @ xyz)
                        + t
                    )

                    rest = values[4:]

                    new_line = (
                        "v "
                        f"{xyz[0]:.9f} "
                        f"{xyz[1]:.9f} "
                        f"{xyz[2]:.9f}"
                    )

                    if rest:
                        new_line += " " + " ".join(rest)

                    fout.write(new_line + "\n")

                    vertex_count += 1

                else:
                    fout.write(line)

            elif line.startswith("vn "):

                values = line.split()

                if len(values) >= 4:

                    n = np.array([
                        float(values[1]),
                        float(values[2]),
                        float(values[3])
                    ])

                    n = R @ n

                    length = np.linalg.norm(n)

                    if length > 0:
                        n /= length

                    fout.write(
                        "vn "
                        f"{n[0]:.9f} "
                        f"{n[1]:.9f} "
                        f"{n[2]:.9f}\n"
                    )

                    normal_count += 1

                else:
                    fout.write(line)

            else:
                # vt, f, mtllib, usemtl etc.
                # remain exactly unchanged.
                fout.write(line)

    print("Saved OBJ :", dst)
    print("Vertices  :", vertex_count)
    print("Normals   :", normal_count)


def main():

    parser = argparse.ArgumentParser(
        description=
        "Apply COLMAP->ENU similarity transform "
        "to PLY or OBJ geometry."
    )

    parser.add_argument(
        "input_mesh"
    )

    parser.add_argument(
        "georef_json"
    )

    parser.add_argument(
        "output_mesh"
    )

    args = parser.parse_args()

    src = Path(args.input_mesh)
    transform_file = Path(args.georef_json)
    dst = Path(args.output_mesh)

    if not src.exists():
        raise FileNotFoundError(src)

    if not transform_file.exists():
        raise FileNotFoundError(transform_file)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scale, R, t = load_transform(
        transform_file
    )

    print("========================================")
    print("GEOREFERENCE TRANSFORM")
    print("========================================")

    print("Scale:", scale)
    print()
    print("Rotation:")
    print(R)
    print()
    print("Translation:", t)
    print()
    print("det(R):", np.linalg.det(R))
    print()

    suffix = src.suffix.lower()

    if suffix == ".ply":

        transform_ply(
            src,
            dst,
            scale,
            R,
            t
        )

    elif suffix == ".obj":

        transform_obj(
            src,
            dst,
            scale,
            R,
            t
        )

    else:
        raise RuntimeError(
            "Supported formats: .ply and .obj"
        )

    print()
    print("========================================")
    print("GEOREFERENCE COMPLETE")
    print("========================================")
    print("Output:", dst.resolve())


if __name__ == "__main__":
    main()
