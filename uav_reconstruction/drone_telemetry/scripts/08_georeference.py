#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def quaternion_to_rotation(qw, qx, qy, qz):

    q = np.array(
        [qw, qx, qy, qz],
        dtype=float
    )

    q /= np.linalg.norm(q)

    w, x, y, z = q

    return np.array([
        [
            1 - 2*y*y - 2*z*z,
            2*x*y - 2*z*w,
            2*x*z + 2*y*w
        ],
        [
            2*x*y + 2*z*w,
            1 - 2*x*x - 2*z*z,
            2*y*z - 2*x*w
        ],
        [
            2*x*z - 2*y*w,
            2*y*z + 2*x*w,
            1 - 2*x*x - 2*y*y
        ]
    ])


def parse_colmap_images(images_txt):

    rows = []

    with open(images_txt, "r") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        values = line.split()

        if len(values) < 10:
            i += 1
            continue

        image_id = int(values[0])

        qw, qx, qy, qz = map(
            float,
            values[1:5]
        )

        tx, ty, tz = map(
            float,
            values[5:8]
        )

        camera_id = int(values[8])

        name = values[9]

        R = quaternion_to_rotation(
            qw, qx, qy, qz
        )

        t = np.array(
            [tx, ty, tz]
        )

        # COLMAP stores:
        # x_camera = R * x_world + t
        #
        # Camera center:
        # C = -R^T t

        C = -R.T @ t

        rows.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "frame_name": name,
            "colmap_x": C[0],
            "colmap_y": C[1],
            "colmap_z": C[2],
        })

        # Skip following POINTS2D line
        i += 2

    return pd.DataFrame(rows)


def gps_to_enu(lat, lon, height):

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    height = np.asarray(height, dtype=float)

    lat0 = np.nanmedian(lat)
    lon0 = np.nanmedian(lon)
    h0 = np.nanmedian(height)

    # WGS84
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    phi0 = np.deg2rad(lat0)

    sin_phi = np.sin(phi0)

    N = a / np.sqrt(
        1.0 - e2 * sin_phi**2
    )

    M = (
        a * (1.0 - e2)
        /
        (1.0 - e2 * sin_phi**2)**1.5
    )

    dlat = np.deg2rad(
        lat - lat0
    )

    dlon = np.deg2rad(
        lon - lon0
    )

    east = (
        dlon *
        (N + h0) *
        np.cos(phi0)
    )

    north = (
        dlat *
        (M + h0)
    )

    up = height - h0

    enu = np.column_stack(
        [east, north, up]
    )

    origin = {
        "latitude": float(lat0),
        "longitude": float(lon0),
        "reference_height_m": float(h0),
    }

    return enu, origin


def umeyama(src, dst):

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    n = len(src)

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)

    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    covariance = (
        dst_centered.T @ src_centered
    ) / n

    U, D, Vt = np.linalg.svd(
        covariance
    )

    S = np.eye(3)

    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    variance_src = np.mean(
        np.sum(
            src_centered ** 2,
            axis=1
        )
    )

    scale = (
        np.trace(
            np.diag(D) @ S
        )
        / variance_src
    )

    translation = (
        mean_dst -
        scale * (R @ mean_src)
    )

    return scale, R, translation


def transform_points(points, scale, R, t):

    return (
        scale *
        (R @ points.T).T
        + t
    )


def error_statistics(pred, target):

    residual = pred - target

    error3d = np.linalg.norm(
        residual,
        axis=1
    )

    horizontal = np.linalg.norm(
        residual[:, :2],
        axis=1
    )

    vertical = np.abs(
        residual[:, 2]
    )

    return {
        "rmse_3d_m":
            float(
                np.sqrt(
                    np.mean(error3d**2)
                )
            ),

        "horizontal_rmse_m":
            float(
                np.sqrt(
                    np.mean(horizontal**2)
                )
            ),

        "vertical_rmse_m":
            float(
                np.sqrt(
                    np.mean(vertical**2)
                )
            ),

        "median_3d_error_m":
            float(np.median(error3d)),

        "p95_3d_error_m":
            float(np.percentile(error3d, 95)),
    }, error3d


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "colmap_model"
    )

    parser.add_argument(
        "frame_gps_csv"
    )

    parser.add_argument(
        "output_dir"
    )

    args = parser.parse_args()

    model = Path(args.colmap_model)
    gps_csv = Path(args.frame_gps_csv)
    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    text_model = out / "colmap_model_txt"

    text_model.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Converting COLMAP model to TXT...")

    subprocess.run([
        "colmap",
        "model_converter",
        "--input_path",
        str(model),
        "--output_path",
        str(text_model),
        "--output_type",
        "TXT"
    ], check=True)

    cameras = parse_colmap_images(
        text_model / "images.txt"
    )

    gps = pd.read_csv(
        gps_csv
    )

    matched = cameras.merge(
        gps,
        on="frame_name",
        how="inner"
    )

    matched = matched.dropna(
        subset=[
            "latitude",
            "longitude",
            "height_m",
            "colmap_x",
            "colmap_y",
            "colmap_z"
        ]
    ).reset_index(drop=True)

    print()
    print("COLMAP cameras:", len(cameras))
    print("GPS rows      :", len(gps))
    print("Matched       :", len(matched))

    if len(matched) < 3:
        raise RuntimeError(
            "Need at least 3 GPS/COLMAP matches."
        )

    enu, origin = gps_to_enu(
        matched["latitude"],
        matched["longitude"],
        matched["height_m"]
    )

    colmap_xyz = matched[
        [
            "colmap_x",
            "colmap_y",
            "colmap_z"
        ]
    ].to_numpy()

    # Initial fit
    scale0, R0, t0 = umeyama(
        colmap_xyz,
        enu
    )

    pred0 = transform_points(
        colmap_xyz,
        scale0,
        R0,
        t0
    )

    _, error0 = error_statistics(
        pred0,
        enu
    )

    median = np.median(error0)

    mad = np.median(
        np.abs(error0 - median)
    )

    sigma = 1.4826 * mad

    threshold = (
        median + 3.0 * sigma
    )

    if sigma == 0:
        inliers = np.ones(
            len(error0),
            dtype=bool
        )
    else:
        inliers = error0 <= threshold

    print()
    print(
        "Initial scale:",
        scale0
    )

    print(
        "MAD threshold:",
        threshold,
        "m"
    )

    print(
        "Inliers:",
        int(inliers.sum()),
        "/",
        len(inliers)
    )

    # Robust refit
    scale, R, t = umeyama(
        colmap_xyz[inliers],
        enu[inliers]
    )

    pred = transform_points(
        colmap_xyz,
        scale,
        R,
        t
    )

    stats, error = error_statistics(
        pred,
        enu
    )

    matched[
        "gps_east_m"
    ] = enu[:, 0]

    matched[
        "gps_north_m"
    ] = enu[:, 1]

    matched[
        "gps_up_m"
    ] = enu[:, 2]

    matched[
        "aligned_east_m"
    ] = pred[:, 0]

    matched[
        "aligned_north_m"
    ] = pred[:, 1]

    matched[
        "aligned_up_m"
    ] = pred[:, 2]

    matched[
        "alignment_error_m"
    ] = error

    matched[
        "alignment_inlier"
    ] = inliers

    matched.to_csv(
        out / "colmap_gps_enu_aligned.csv",
        index=False
    )

    cameras.to_csv(
        out / "colmap_camera_centers.csv",
        index=False
    )

    result = {
        "coordinate_system":
            "ENU: X=East, Y=North, Z=Up",

        "height_note":
            "Up uses relative telemetry height, not sea-level geodetic elevation.",

        "origin": origin,

        "scale_m_per_colmap_unit":
            float(scale),

        "rotation_matrix":
            R.tolist(),

        "translation_m":
            t.tolist(),

        "det_rotation":
            float(np.linalg.det(R)),

        "matched_cameras":
            int(len(matched)),

        "inlier_cameras":
            int(inliers.sum()),

        "mad_outlier_threshold_m":
            float(threshold),

        **stats
    }

    with open(
        out / "georef_similarity.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("========================================")
    print("GEOREFERENCE FIT COMPLETE")
    print("========================================")

    print(
        "Scale:",
        scale,
        "m/COLMAP unit"
    )

    print(
        "det(R):",
        np.linalg.det(R)
    )

    for key, value in stats.items():
        print(
            f"{key}: {value:.4f}"
        )

    print()
    print(
        "Transform:",
        out / "georef_similarity.json"
    )

    print(
        "Aligned cameras:",
        out / "colmap_gps_enu_aligned.csv"
    )


if __name__ == "__main__":
    main()
