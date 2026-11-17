"""Write a standalone viewer for a completed recorded-video sparse map."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .session_reconstruct import read_model


def write_recorded_viewer(output):
    output = Path(output).resolve()
    state = json.loads((output / 'checkpoint.json').read_text())
    online = state['online']
    model = online.get('best_text')
    if not model:
        raise RuntimeError('No accepted sparse model is available for the viewer')
    _, poses, points = read_model(Path(model))
    step = max(1, int(np.ceil(len(points) / 30000)))
    xyzrgb = [point['xyz'] + point.get('rgb', [180, 180, 180]) for point in points[::step]]
    trajectory = []
    for pose in poses:
        w, x, y, z = pose['quaternion_wxyz']
        rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                             [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                             [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
        trajectory.append((-rotation.T @ np.asarray(pose['translation_world_to_camera'])).tolist())
    accepted = [update for update in online.get('map_updates', []) if update.get('accepted')]
    frame_file = output / 'images' / online['keyframes'][-1]
    frame = 'data:image/jpeg;base64,' + base64.b64encode(frame_file.read_bytes()).decode()
    status = {'state': 'complete', 'source_label': 'Saved video frame',
              'captured_frames': online['sampled_frames'], 'processed_frames': online['sampled_frames'],
              'buffered_frames': 0, 'keyframes': len(online['keyframes']),
              'registered_cameras': len(poses), 'sparse_points': len(points),
              'accepted_updates': len(accepted),
              'reprojection_error_px': accepted[-1].get('median_reprojection_error_px') if accepted else None,
              'imu_samples': 0, 'imu_rate_hz': None, 'gps': {}, 'camera': {},
              'error': None, 'message': 'Causal sparse reconstruction of a saved video',
              'calibration_exact': state['config'].get('calibration_exact', False),
              'frame_clock_source': 'recorded_video_time', 'rotation_compensated': False,
              'last_selection_reason': None}
    map_data = {'points': xyzrgb, 'trajectory': trajectory,
                'revision': len(online.get('map_updates', []))}

    def javascript(value):
        return json.dumps(value, separators=(',', ':')).replace('<', '\\u003c')

    prefix = '<script>\n' + ''.join(f'window.{name}={javascript(value)};\n' for name, value in
        (('FINAL_MAP', map_data), ('FINAL_STATUS', status), ('FINAL_FRAME', frame))) + '</script>\n'
    template = Path(__file__).with_name('phone_dashboard.html').read_text()
    page = template.replace('<script>\n', prefix + '<script>\n', 1)
    path = output / 'viewer.html'
    path.write_text(page)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reconstruction', type=Path)
    print(write_recorded_viewer(parser.parse_args().reconstruction))


if __name__ == '__main__':
    main()
