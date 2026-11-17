"""Bounded-memory native-rate AMtown geometry for the isolated expansion.

Uses the existing published AMtown extrinsic and released reference trajectory.
Other new sources remain appearance-only until their geometry is qualified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .lidar_depth_supervision import extrinsics_for_scene
from .prepare_multisource_selector import cloud, emit, project, write


def geometry(job):
    task, output = job
    output = Path(output)
    cv2.setNumThreads(1)
    flight = task['flight']
    if flight not in ('AMtown01', 'AMtown02') or task['role'] != 'train':
        raise ValueError('Only already supported AMtown training calibration is admitted')
    marker = output / f'{flight}_native_geometry.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get('state') == 'complete' and old['identity'] == task:
            return old
    r, t = extrinsics_for_scene(flight)
    rows, paths, exclusions, visited, used = [], [], [], 0, 0
    with zipfile.ZipFile(task['path']) as archive:
        names = archive.namelist()
        prefix = 'interval1_' + flight + '/'
        infos = json.loads(archive.read(prefix+'sampleinfos_interpolated.json'))
        by_name = {row['OriginalImageName']:row for row in infos}
        ordered = sorted(infos, key=lambda row: float(Path(row['OriginalImageName']).stem))
        times = np.array([float(Path(row['OriginalImageName']).stem) for row in ordered])
        if np.any(np.diff(times) <= 0):
            raise ValueError('Nonmonotonic/duplicate reference timestamps')
        poses = np.array([row['T4x4'] for row in ordered], dtype=float)
        rotation = Slerp(times, Rotation.from_matrix(poses[:,:3,:3]))
        scans = {}
        for name in names:
            match = re.search(r'/interval1_LIDAR/image([0-9.]+)_lidar([0-9.]+)\.txt$', name)
            if match:
                if match[1] in scans:
                    raise ValueError('Ambiguous synchronized scan')
                scans[match[1]] = (name, float(match[2]))
        images = sorted(n for n in names if n.startswith(prefix+'interval1_CAM/') and n.endswith('.jpg'))
        for index, name in enumerate(images):
            visited += 1
            key = Path(name).stem
            if Path(name).name not in by_name or key not in scans:
                exclusions.append(dict(index=index, reason='missing paired pose/scan'))
                continue
            info = by_name[Path(name).name]
            scan, lidar_time = scans[key]
            if abs(float(key)-lidar_time) > .05 or not times[0] <= lidar_time <= times[-1]:
                exclusions.append(dict(index=index, reason='scan outside 50ms/trajectory support'))
                continue
            position = np.searchsorted(times, lidar_time)
            if 0 < position < len(times) and times[position]-times[position-1] > 1.:
                exclusions.append(dict(index=index, reason='pose interpolation gap exceeds 1s'))
                continue
            if any(abs(float(info[k])) > 1e-12 for k in ('K1','K2','K3','P1','P2')):
                raise ValueError('Released map RGB unexpectedly has nonzero distortion')
            image = cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (info['Height'], info['Width']):
                raise ValueError('Native image/calibration size mismatch')
            xyz = np.loadtxt(io.BytesIO(archive.read(scan)),dtype=np.float64).reshape(-1,3)
            xyz = xyz[np.isfinite(xyz).all(1)]
            W_C = np.eye(4)
            W_C[:3,:3] = rotation([lidar_time]).as_matrix()[0]
            W_C[:3,3] = [np.interp(lidar_time,times,poses[:,axis,3]) for axis in range(3)]
            at_lidar = xyz @ r.T + t
            points = at_lidar @ W_C[:3,:3].T + W_C[:3,3]
            camera_world = np.asarray(info['T4x4'], dtype=float)
            P = np.linalg.inv(camera_world)
            K = np.asarray(info['P3x3'],dtype=float)
            K[0] *= 256 / info['Width']; K[1] *= 192 / info['Height']
            points = cloud(project(points,P,K),K,camera_world)
            if len(points) < 100:
                exclusions.append(dict(index=index, reason='fewer than 100 in-view measured pixels'))
                continue
            rgb = cv2.cvtColor(cv2.resize(image,(256,192),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            rows.append(dict(rgb=rgb,K=K,P=P,points=points,index=index));used += 1
            if len(rows) == 32:
                paths.extend(emit(output,flight+f'_native_at{index:06d}','uavscenes','train',rows,
                    'released_AMtown_LiDAR_published_extrinsic_interpolated_reference_pose_static_scene'))
                rows = []
            if visited % 100 == 0:
                write(output/f'{flight}_native_progress.json',dict(state='running',visited=visited,
                      total=len(images),usable_before_episode_gates=used,episodes=len(paths),excluded=len(exclusions)))
        if rows:
            paths.extend(emit(output,flight+'_native_tail','uavscenes','train',rows,
                'released_AMtown_LiDAR_published_extrinsic_interpolated_reference_pose_static_scene'))
    if not paths:
        raise ValueError('No native-rate episodes passed the reconstruction-supervision gates')
    result = dict(state='complete',identity=task,visited=visited,usable_before_episode_gates=used,
                  episodes=len(paths),excluded=exclusions,
                  limits='Reference-pose/static-scene proxy. No deskew or dynamic-object guarantee; '
                         'spatial-cell-disjoint anchors/reference; not end-to-end mesh evaluation.')
    write(marker,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion',type=Path,required=True)
    args=parser.parse_args();out=args.expansion/'training';(out/'geometry').mkdir(parents=True,exist_ok=True)
    tasks=[t for t in json.loads((args.expansion/'admission_plan.json').read_text())['tasks']
           if t.get('flight') in ('AMtown01','AMtown02') and t['kind']=='uav_zip']
    if len(tasks)!=2:raise ValueError('Expected both complete native-rate AMtown training flights')
    write(out/'geometry_status.json',dict(state='running',total=2,completed=0))
    done=[];failures=[]
    with ProcessPoolExecutor(2) as pool:
        pending={pool.submit(geometry,(t,str(out))):t['flight'] for t in tasks}
        for future in as_completed(pending):
            try:
                result=future.result();done.append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('identity','excluded')}),flush=True)
            except Exception as exc:
                failures.append(dict(flight=pending[future],error=repr(exc)));print(failures[-1],flush=True)
            write(out/'geometry_status.json',dict(state='running',total=2,completed=len(done),failures=failures))
    write(out/'geometry_status.json',dict(state='failed' if failures else 'complete',total=2,
          completed=len(done),failures=failures,episodes=sum(r['episodes'] for r in done)))
    if failures:raise RuntimeError('Native-rate geometry preparation failed; original jobs are unchanged')


if __name__=='__main__':main()
