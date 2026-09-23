"""Inspect validation archive structure and metric-array representation."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import sys
import tarfile
import time
import h5py
import numpy as np


def main(directory, acquisition):
    started = time.perf_counter()
    records = []
    examples = []
    for archive_name in ("validation_000000.tar", "validation_000001.tar"):
        with tarfile.open(acquisition / archive_name, "r:") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                assert not relative.is_absolute() and ".." not in relative.parts
                if not member.isfile() or relative.suffix != ".h5":
                    continue
                records.append({"archive": archive_name, "member": member.name,
                                "bytes": member.size, "offset": member.offset_data})
            available = [record for record in records if record["archive"] == archive_name]
            for record in available[:3]:
                data = archive.extractfile(record["member"]).read()
                with h5py.File(io.BytesIO(data), "r") as source:
                    arrays = {name: np.asarray(source[name]) for name in source.keys()}
                examples.append(record | {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                                      "minimum": float(np.nanmin(array)),
                                      "maximum": float(np.nanmax(array)),
                                      "finite_fraction": float(np.isfinite(array).mean())}
                               for name, array in arrays.items()},
                })
    (directory / "members.json").write_text(json.dumps(records, indent=2) + "\n")
    summary = {"status": "complete", "member_count": len(records),
               "examples": examples, "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
