"""Read a fixed HDF5 archive member without extracting archive paths."""
import hashlib
import io
import h5py
import numpy as np


def read_frame(acquisition, record):
    with (acquisition / record["archive"]).open("rb") as source:
        source.seek(record["offset"])
        data = source.read(record["bytes"])
    assert len(data) == record["bytes"]
    digest = hashlib.sha256(data).hexdigest()
    if "sha256" in record:
        assert digest == record["sha256"]
    with h5py.File(io.BytesIO(data), "r") as source:
        rgb = np.asarray(source["rgb"])
        depth = np.asarray(source["depth"])
    assert rgb.shape == (3, 480, 640) and rgb.dtype == np.uint8
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    assert np.isfinite(depth).all()
    return rgb.transpose(1, 2, 0), depth, digest
