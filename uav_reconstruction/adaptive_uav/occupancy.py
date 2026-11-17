"""Sparse bounded log-odds occupancy, distinct from a fused point cloud.

Absent cells are unknown. Invalid/no-return/dynamic pixels must be skipped by
the caller, never integrated as long empty rays. Log odds are a model, not a
calibrated statement about physical safety.
"""
from collections import OrderedDict
import json,sqlite3,tempfile,os
from pathlib import Path
import numpy as np


def traverse(origin,endpoint,voxel_size):
    """Amanatides-Woo grid traversal, including both origin and endpoint cells."""
    origin=np.asarray(origin,float);endpoint=np.asarray(endpoint,float)
    if not np.isfinite([origin,endpoint]).all() or voxel_size<=0:raise ValueError('Finite ray and positive voxel size required')
    cell=np.floor(origin/voxel_size).astype(int);last=np.floor(endpoint/voxel_size).astype(int)
    if np.abs(last-cell).sum()>10000:raise ValueError('Ray exceeds bounded traversal budget')
    yield tuple(cell)
    if np.array_equal(cell,last):return
    direction=endpoint-origin;step=np.sign(direction).astype(int)
    delta=np.full(3,np.inf);moving=direction!=0;delta[moving]=voxel_size/np.abs(direction[moving])
    boundary=(cell+(step>0))*voxel_size;tmax=np.full(3,np.inf)
    tmax[moving]=(boundary[moving]-origin[moving])/direction[moving]
    for _ in range(int(np.abs(last-cell).sum())+3):
        if np.array_equal(cell,last):return
        # Ties advance all tied axes to avoid marking side-adjacent cells free
        # when the ray crosses exactly a corner.
        axes=np.isclose(tmax,np.min(tmax),rtol=0,atol=1e-12)
        cell[axes]+=step[axes];tmax[axes]+=delta[axes]
        yield tuple(cell)
    raise RuntimeError('Ray traversal failed to reach endpoint')


class OccupancyMap:
    def __init__(self,voxel_size=.1,max_cells=100000,free_log_odds=-.4,occupied_log_odds=.85):
        if voxel_size<=0 or max_cells<1:raise ValueError('Positive occupancy budgets required')
        self.voxel_size=voxel_size;self.max_cells=max_cells;self.free=free_log_odds;self.occupied=occupied_log_odds
        self.cells=OrderedDict()
        handle,path=tempfile.mkstemp(prefix='geostream_observations_',suffix='.sqlite');os.close(handle)
        self.database_path=Path(path);self.database=sqlite3.connect(path)
        self.database.execute('CREATE TABLE observations (id TEXT PRIMARY KEY)')

    def integrate(self,origin,endpoint,observation_id,confidence=1.,valid=True,excluded=False):
        if not valid or excluded:return False
        if not np.isfinite(confidence) or not 0<confidence<=1:return False
        cells=list(traverse(origin,endpoint,self.voxel_size))
        cursor=self.database.execute('INSERT OR IGNORE INTO observations VALUES (?)',(str(observation_id),))
        self.database.commit()
        if not cursor.rowcount:return False
        for index,cell in enumerate(cells):
            update=self.occupied if index==len(cells)-1 else self.free
            old=self.cells.pop(cell,0.);self.cells[cell]=float(np.clip(old+confidence*update,-4,4))
            if len(self.cells)>self.max_cells:self.cells.popitem(last=False)
        return True

    def state(self,cell):
        value=self.cells.get(tuple(cell))
        if value is None:return 'unknown'
        if value>.5:return 'occupied'
        if value<-.2:return 'free'
        return 'observed_uncertain'

    def save(self,path):
        sidecar=Path(path).with_suffix('.observations.sqlite')
        with sqlite3.connect(sidecar) as target:self.database.backup(target)
        Path(path).write_text(json.dumps({'voxel_size':self.voxel_size,'max_cells':self.max_cells,
            'free_log_odds':self.free,'occupied_log_odds':self.occupied,
            'cells':[[*map(int,key),value] for key,value in self.cells.items()],'observation_database':sidecar.name}))

    @classmethod
    def load(cls,path):
        state=json.loads(Path(path).read_text());result=cls(**{k:state[k] for k in ('voxel_size','max_cells','free_log_odds','occupied_log_odds')})
        result.cells=OrderedDict((tuple(row[:3]),row[3]) for row in state['cells'])
        with sqlite3.connect(Path(path).parent/state['observation_database']) as source:source.backup(result.database)
        return result

    def close(self):
        self.database.close();self.database_path.unlink(missing_ok=True)
