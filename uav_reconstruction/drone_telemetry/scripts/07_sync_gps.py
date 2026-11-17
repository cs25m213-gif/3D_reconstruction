#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def find_column(df, candidates, required=True):
    cols = list(df.columns)

    # Exact match first
    for c in candidates:
        if c in cols:
            return c

    # Case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    # Substring fallback
    for candidate in candidates:
        for col in cols:
            if candidate.lower() in col.lower():
                return col

    if required:
        raise RuntimeError(
            f"Could not find column matching {candidates}\n"
            f"Available columns:\n{cols}"
        )

    return None


def to_bool(series):
    true_values = {
        "1", "true", "yes", "on", "recording"
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(true_values)
        .to_numpy()
    )


def video_duration(video):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video)
    ]

    return float(
        subprocess.check_output(cmd, text=True).strip()
    )


def contiguous_segments(mask):
    segments = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        if start is not None and (
            not value or i == len(mask) - 1
        ):
            end = i if value else i - 1
            segments.append((start, end))
            start = None

    return segments


def main():

    parser = argparse.ArgumentParser(
        description="Synchronize extracted video frames with DJI telemetry."
    )

    parser.add_argument("video")
    parser.add_argument("telemetry_csv")
    parser.add_argument("frames_dir")
    parser.add_argument("output_csv")

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--timezone",
        default="America/Los_Angeles"
    )

    args = parser.parse_args()

    video = Path(args.video)
    telemetry = Path(args.telemetry_csv)
    frames_dir = Path(args.frames_dir)
    output = Path(args.output_csv)

    duration = video_duration(video)

    print("Video duration:", duration, "s")

    df = pd.read_csv(telemetry)

    print("Telemetry rows:", len(df))

    time_col = find_column(
        df,
        ["timestamps_ns", "timestamp", "unix"]
    )

    video_col = find_column(
        df,
        [
            "CAMERA.isVideo",
            "isVideo",
            "camera.isvideo"
        ]
    )

    lat_col = find_column(
        df,
        ["OSD.latitude", "latitude"]
    )

    lon_col = find_column(
        df,
        ["OSD.longitude", "longitude"]
    )

    height_col = find_column(
        df,
        ["OSD.height [ft]", "OSD.height", "height"]
    )

    altitude_col = find_column(
        df,
        ["OSD.altitude [ft]", "OSD.altitude", "altitude"],
        required=False
    )

    fly_col = find_column(
        df,
        ["OSD.flyTime [s]", "flyTime"],
        required=False
    )

    sat_col = find_column(
        df,
        ["GPS.numSatellites", "numSatellites", "satellites"],
        required=False
    )

    gps_used_col = find_column(
        df,
        ["GPS.isUsed", "GPS.used", "gps used"],
        required=False
    )

    pitch_col = find_column(
        df,
        ["OSD.pitch", "pitch"],
        required=False
    )

    roll_col = find_column(
        df,
        ["OSD.roll", "roll"],
        required=False
    )

    yaw_col = find_column(
        df,
        ["OSD.yaw", "yaw"],
        required=False
    )

    gimbal_pitch_col = find_column(
        df,
        ["GIMBAL.pitch", "gimbal pitch"],
        required=False
    )

    gimbal_roll_col = find_column(
        df,
        ["GIMBAL.roll", "gimbal roll"],
        required=False
    )

    gimbal_yaw_col = find_column(
        df,
        ["GIMBAL.yaw", "gimbal yaw"],
        required=False
    )

    ts = pd.to_numeric(
        df[time_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    # Some DJI CSV exporters call the column timestamps_ns
    # even when values are already Unix seconds.
    med = np.nanmedian(ts)

    if med > 1e17:
        ts = ts / 1e9
    elif med > 1e14:
        ts = ts / 1e6
    elif med > 1e11:
        ts = ts / 1e3

    recording = to_bool(df[video_col])

    segments = contiguous_segments(recording)

    if not segments:
        raise RuntimeError(
            "No CAMERA.isVideo recording segment found."
        )

    print("\nRecording segments:")

    scored = []

    for start, end in segments:

        segment_duration = ts[end] - ts[start]

        difference = abs(
            segment_duration - duration
        )

        scored.append(
            (difference, start, end, segment_duration)
        )

        print(
            f"rows {start}..{end} "
            f"duration={segment_duration:.3f}s "
            f"video_difference={difference:.3f}s"
        )

    scored.sort()

    _, start, end, segment_duration = scored[0]

    record_start = ts[start]

    print()
    print("Selected segment:")
    print(" start row :", start)
    print(" end row   :", end)
    print(" duration  :", segment_duration)
    print(" video     :", duration)
    print(" unix start:", record_start)

    frames = sorted(
        list(frames_dir.glob("*.jpg")) +
        list(frames_dir.glob("*.jpeg")) +
        list(frames_dir.glob("*.png"))
    )

    if not frames:
        raise RuntimeError(
            f"No frames found in {frames_dir}"
        )

    print("Frames:", len(frames))

    video_times = (
        np.arange(len(frames), dtype=float)
        / args.fps
    )

    query_times = record_start + video_times

    segment_mask = (
        (ts >= ts[start]) &
        (ts <= ts[end])
    )

    seg_ts = ts[segment_mask]

    def interpolate(col):
        if col is None:
            return np.full(
                len(query_times),
                np.nan
            )

        values = pd.to_numeric(
            df.loc[segment_mask, col],
            errors="coerce"
        ).to_numpy(dtype=float)

        good = (
            np.isfinite(seg_ts) &
            np.isfinite(values)
        )

        if good.sum() < 2:
            return np.full(
                len(query_times),
                np.nan
            )

        return np.interp(
            query_times,
            seg_ts[good],
            values[good]
        )

    latitude = interpolate(lat_col)
    longitude = interpolate(lon_col)

    height_ft = interpolate(height_col)

    altitude_ft = (
        interpolate(altitude_col)
        if altitude_col
        else np.full(len(frames), np.nan)
    )

    fly_time = interpolate(fly_col)

    satellites = interpolate(sat_col)

    drone_pitch = interpolate(pitch_col)
    drone_roll = interpolate(roll_col)
    drone_yaw = interpolate(yaw_col)

    gimbal_pitch = interpolate(gimbal_pitch_col)
    gimbal_roll = interpolate(gimbal_roll_col)
    gimbal_yaw = interpolate(gimbal_yaw_col)

    # Boolean GPS-used flag: nearest telemetry sample.
    if gps_used_col:

        gps_raw = to_bool(
            df.loc[segment_mask, gps_used_col]
        )

        idx = np.searchsorted(
            seg_ts,
            query_times
        )

        idx = np.clip(
            idx,
            0,
            len(seg_ts) - 1
        )

        gps_used = gps_raw[idx]

    else:
        gps_used = np.full(
            len(frames),
            True
        )

    timezone = ZoneInfo(args.timezone)

    utc_dt = pd.to_datetime(
        query_times,
        unit="s",
        utc=True
    )

    local_dt = [
        x.to_pydatetime()
        .astimezone(timezone)
        .isoformat()
        for x in utc_dt
    ]

    output_df = pd.DataFrame({
        "frame_name":
            [p.name for p in frames],

        "video_time_s":
            video_times,

        "telemetry_unix_s":
            query_times,

        "timestamp_utc":
            utc_dt.astype(str),

        "timestamp_local":
            local_dt,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "height_ft":
            height_ft,

        "height_m":
            height_ft * 0.3048,

        "altitude_ft":
            altitude_ft,

        "altitude_m":
            altitude_ft * 0.3048,

        "flight_time_s":
            fly_time,

        "gps_satellites":
            satellites,

        "gps_used":
            gps_used,

        "drone_pitch":
            drone_pitch,

        "drone_roll":
            drone_roll,

        "drone_yaw":
            drone_yaw,

        "gimbal_pitch":
            gimbal_pitch,

        "gimbal_roll":
            gimbal_roll,

        "gimbal_yaw":
            gimbal_yaw,
    })

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        output,
        index=False
    )

    print()
    print("========================================")
    print("GPS SYNC COMPLETE")
    print("========================================")
    print("Output:", output.resolve())
    print("Rows:", len(output_df))
    print(
        "Valid GPS:",
        output_df[
            ["latitude", "longitude"]
        ].notna().all(axis=1).sum()
    )


if __name__ == "__main__":
    main()
