"""Compatible NPZ depth caches with less CPU time spent on compression."""
import os
import zipfile

import numpy as np


def save_depth_cache(path, **arrays):
    """Match savez_compressed's arrays and container, using DEFLATE level 1.

    Atomic publication remains the caller's existing temporary-file rename.
    Level 6 can be selected for an operational rollback without changing tensors.
    """
    level = int(os.environ.get('GEOSTREAM_CACHE_COMPRESSION', '1'))
    if not 0 <= level <= 9:
        raise ValueError('GEOSTREAM_CACHE_COMPRESSION must be in 0..9')
    with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=level, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=True)
