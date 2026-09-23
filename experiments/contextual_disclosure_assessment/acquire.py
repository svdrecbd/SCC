"""Acquire only declared files, retaining hashes and a receipt on failure."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


def main(directory):
    configuration = json.loads((directory / "acquisition_config.json").read_text())
    started = time.monotonic()
    receipt = {"files": [], "bytes": 0, "training": False}
    try:
        for specification in configuration["files"]:
            destination = directory / "downloads" / specification["path"]
            assert destination.is_relative_to(directory / "downloads")
            destination.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(specification["url"], headers={"User-Agent": "SCC-research-acquisition"})
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(request, timeout=30) as response, destination.open("xb") as output:
                while data := response.read(65536):
                    receipt["bytes"] += len(data)
                    count += len(data)
                    assert receipt["bytes"] <= configuration["maximum_bytes"]
                    assert time.monotonic() - started <= configuration["wall_seconds"]
                    output.write(data)
                    digest.update(data)
            if "bytes" in specification:
                assert count == specification["bytes"]
            if "sha256" in specification:
                assert digest.hexdigest() == specification["sha256"]
            receipt["files"].append({**specification, "bytes": count, "sha256": digest.hexdigest()})
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        (directory / "acquisition.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({key: value for key, value in receipt.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
