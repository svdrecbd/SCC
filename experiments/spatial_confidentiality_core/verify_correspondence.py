"""Compare fixed original NYU frame indices across the two data sources."""
from pathlib import Path
import io
import json
import sys
import tarfile
import time
import h5py
import numpy as np
from acquire_scene_metadata import RangeFile


def main(directory, acquisition):
    started = time.perf_counter()
    metadata = json.loads((directory / "source_metadata.json").read_text())
    source = RangeFile(metadata["source_url"], metadata["source_file"]["size"], directory)
    records = []
    with h5py.File(source, "r") as original:
        with tarfile.open(acquisition / "validation_000000.tar", "r:") as archive:
            for index in (1, 9):
                member = f"val/official/{index:05d}.h5"
                with h5py.File(io.BytesIO(archive.extractfile(member).read()), "r") as frame:
                    rgb = np.asarray(frame["rgb"])
                    depth = np.asarray(frame["depth"])
                reference_rgb = np.asarray(original["images"][index - 1]).transpose(0, 2, 1)
                reference_depth = np.asarray(original["depths"][index - 1]).T
                rgb_equal = bool(np.array_equal(rgb, reference_rgb))
                maximum_depth_error = float(np.max(np.abs(depth - reference_depth)))
                record = {"index": index, "rgb_equal": rgb_equal,
                          "maximum_depth_error_metres": maximum_depth_error,
                          "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape)}
                records.append(record)
                (directory / "partial_result.json").write_text(json.dumps(records, indent=2) + "\n")
                assert rgb_equal and maximum_depth_error <= 1e-6, record
    summary = {"status": "complete", "records": records,
               "downloaded_bytes": source.downloaded_bytes, "range_count": len(source.records),
               "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
