"""Acquire explicitly listed spatial-model artifacts with byte and digest checks."""
from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request


def main(directory):
    configuration = json.loads((directory / "config.json").read_text())
    started = time.perf_counter()
    records = []
    total_bytes = 0
    status = "incomplete"
    error = None
    try:
        for specification in configuration["files"]:
            destination = directory / specification["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(specification["url"], timeout=20) as response:
                with destination.open("xb") as output:
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        total_bytes += len(block)
                        if total_bytes > configuration["maximum_bytes"]:
                            raise RuntimeError("Acquisition byte limit exceeded")
                        output.write(block)
                        digest.update(block)
            record = specification | {"bytes": count, "sha256": digest.hexdigest()}
            records.append(record)
            if "expected_sha256" in specification:
                assert digest.hexdigest() == specification["expected_sha256"]
            print(json.dumps({"file": specification["destination"], "bytes": count}), flush=True)
        status = "complete"
    except Exception as exception:
        error = repr(exception)
        raise
    finally:
        (directory / "acquisition_receipt.json").write_text(json.dumps({
            "status": status, "error": error, "files": records,
            "total_bytes": total_bytes, "wall_seconds": time.perf_counter() - started,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
