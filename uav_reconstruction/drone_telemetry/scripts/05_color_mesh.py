#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("dense_cloud")
    parser.add_argument("mesh")
    parser.add_argument("output")

    args = parser.parse_args()

    dense_path = Path(args.dense_cloud)
    mesh_path = Path(args.mesh)
    output_path = Path(args.output)

    print("Loading dense RGB point cloud...")
    cloud = o3d.io.read_point_cloud(str(dense_path))

    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not cloud.has_colors():
        raise RuntimeError(
            "Dense point cloud does not contain RGB colors."
        )

    dense_xyz = np.asarray(cloud.points)
    dense_rgb = np.asarray(cloud.colors)

    mesh_xyz = np.asarray(mesh.vertices)

    print()
    print("Dense points :", len(dense_xyz))
    print("Mesh vertices:", len(mesh_xyz))
    print("Mesh faces   :", len(mesh.triangles))

    print("\nBuilding KD-tree...")

    tree = cKDTree(dense_xyz)

    print("Transferring nearest RGB values...")

    distance, index = tree.query(
        mesh_xyz,
        k=1,
        workers=-1
    )

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        dense_rgb[index]
    )

    mesh.compute_vertex_normals()

    print()
    print("Mean distance  :", np.mean(distance))
    print("Median distance:", np.median(distance))
    print("Maximum distance:", np.max(distance))

    print("\nSaving:", output_path)

    ok = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_colors=True
    )

    if not ok:
        raise RuntimeError("Failed to save mesh.")

    print("Done.")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.triangles))


if __name__ == "__main__":
    main()
