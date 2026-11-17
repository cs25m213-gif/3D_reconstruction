"""Explicit WGS84-to-local ENU conversion; conversion does not calibrate scale."""
import numpy as np


def ecef(latitude,longitude,altitude):
    lat,lon=np.radians([latitude,longitude]);a=6378137.;e2=6.69437999014e-3
    n=a/np.sqrt(1-e2*np.sin(lat)**2)
    return np.array([(n+altitude)*np.cos(lat)*np.cos(lon),(n+altitude)*np.cos(lat)*np.sin(lon),
        (n*(1-e2)+altitude)*np.sin(lat)])


def gps_to_enu(points,origin):
    points=np.asarray(points,float);origin=np.asarray(origin,float)
    if points.ndim!=2 or points.shape[1]!=3 or origin.shape!=(3,) or not np.isfinite(points).all() or not np.isfinite(origin).all():
        raise ValueError('Finite latitude/longitude/ellipsoid-altitude triples required')
    if np.any(np.abs(points[:,0])>90) or np.any(np.abs(points[:,1])>180):raise ValueError('Invalid GPS angles')
    lat,lon=np.radians(origin[:2]);rotation=np.array([[-np.sin(lon),np.cos(lon),0],
        [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
        [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    return np.stack([rotation@(ecef(*p)-ecef(*origin)) for p in points])
