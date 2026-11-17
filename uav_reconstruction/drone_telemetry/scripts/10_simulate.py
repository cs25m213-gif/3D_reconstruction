#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project",
        default="/home/vishnu/eswar/drone_telemetry"
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=2.0
    )

    args = ap.parse_args()

    root = Path(args.project)

    work = root / "reconstruction_2fps_human_animal_mask"

    frames_dir = work / "images"

    gps_csv = work / "frame_gps_sync.csv"

    aligned_csv = (
        work
        / "georef"
        / "colmap_gps_enu_aligned.csv"
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading GPS...")
    gps = pd.read_csv(gps_csv)

    print("Loading aligned COLMAP cameras...")
    cams = pd.read_csv(aligned_csv)

    frames = sorted(frames_dir.glob("*.jpg"))

    if not frames:
        raise RuntimeError(
            f"No frames found in {frames_dir}"
        )

    print("Frames:", len(frames))
    print("GPS rows:", len(gps))
    print("Registered cameras:", len(cams))

    # ---------------------------------------------------------
    # Camera trajectory
    # ---------------------------------------------------------

    east = cams["aligned_east_m"].to_numpy()
    north = cams["aligned_north_m"].to_numpy()
    up = cams["aligned_up_m"].to_numpy()

    # frame_name -> camera row
    camera_lookup = {
        row.frame_name: i
        for i, row in cams.iterrows()
    }

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(16, 8))

    ax_img = fig.add_subplot(1, 2, 1)

    ax_3d = fig.add_subplot(
        1,
        2,
        2,
        projection="3d"
    )

    # ---------------------------------------------------------
    # First image
    # ---------------------------------------------------------

    img = cv2.imread(str(frames[0]))

    if img is None:
        raise RuntimeError(
            f"Cannot read {frames[0]}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    image_artist = ax_img.imshow(img)

    ax_img.axis("off")

    title = ax_img.set_title(
        "SegFormer processed frame"
    )

    # ---------------------------------------------------------
    # Full camera path
    # ---------------------------------------------------------

    ax_3d.plot(
        east,
        north,
        up,
        linewidth=1
    )

    marker, = ax_3d.plot(
        [east[0]],
        [north[0]],
        [up[0]],
        marker="o",
        markersize=8
    )

    ax_3d.set_xlabel("East (m)")
    ax_3d.set_ylabel("North (m)")
    ax_3d.set_zlabel("Up (m)")

    ax_3d.set_title(
        "GPS-aligned COLMAP camera trajectory"
    )

    # ---------------------------------------------------------
    # Telemetry box
    # ---------------------------------------------------------

    info_text = fig.text(
        0.02,
        0.02,
        "",
        fontsize=10
    )

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    last_cam_index = 0

    def update(frame_index):

        nonlocal last_cam_index

        frame_path = frames[frame_index]

        image = cv2.imread(
            str(frame_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_artist.set_data(image)

        name = frame_path.name

        # Exact COLMAP registration if available
        if name in camera_lookup:

            last_cam_index = camera_lookup[name]

        i = last_cam_index

        marker.set_data_3d(
            [east[i]],
            [north[i]],
            [up[i]]
        )

        if frame_index < len(gps):

            row = gps.iloc[frame_index]

            lat = row.get(
                "latitude",
                np.nan
            )

            lon = row.get(
                "longitude",
                np.nan
            )

            height = row.get(
                "height_m",
                np.nan
            )

            video_time = row.get(
                "video_time_s",
                frame_index / args.fps
            )

            info_text.set_text(
                f"Frame: {name}\n"
                f"Video time: {video_time:.2f} s\n"
                f"Latitude: {lat:.7f}\n"
                f"Longitude: {lon:.7f}\n"
                f"Relative height: {height:.2f} m\n"
                f"ENU camera: "
                f"E={east[i]:.2f}  "
                f"N={north[i]:.2f}  "
                f"U={up[i]:.2f}"
            )

            title.set_text(
                f"SegFormer frame — {video_time:.1f} s"
            )

        return (
            image_artist,
            marker,
            info_text,
            title
        )

    # 2 FPS -> 500 ms between frames

    interval_ms = 1000.0 / args.fps

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    plt.show()


if __name__ == "__main__":
    main()
