#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("output")

    parser.add_argument(
        "--no-y-up",
        action="store_true",
        help="Do not convert reconstruction Z-up to glTF Y-up"
    )

    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    print("Loading:", src)

    loaded = trimesh.load(
        src,
        force="scene",
        process=False
    )

    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded

    print("Geometries:", len(scene.geometry))

    total_vertices = 0
    total_faces = 0

    # Reconstruction:
    # X, Y, Z-up
    #
    # glTF:
    # X' = X
    # Y' = Z
    # Z' = -Y

    transform = np.array([
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ])

    for name, mesh in scene.geometry.items():

        print()
        print("Geometry:", name)

        if hasattr(mesh, "vertices"):
            print("Vertices:", len(mesh.vertices))
            total_vertices += len(mesh.vertices)

        if hasattr(mesh, "faces"):
            print("Faces:", len(mesh.faces))
            total_faces += len(mesh.faces)

        if not args.no_y_up:
            mesh.apply_transform(transform)

    print("\nExporting...")

    dst.write_bytes(
        scene.export(file_type="glb")
    )

    print()
    print("Created:", dst.resolve())
    print("Vertices:", total_vertices)
    print("Faces:", total_faces)
    print(
        "Size:",
        round(dst.stat().st_size / 1024**2, 2),
        "MB"
    )


if __name__ == "__main__":
    main()
