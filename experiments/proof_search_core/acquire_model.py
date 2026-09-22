"""Acquire a public model at a fixed revision and verify every weight shard."""
import hashlib
import json
import sys
import time
from pathlib import Path
import requests


def main(directory):
    configuration = json.loads((directory / "large_model_config.json").read_text())
    destination = directory / "model"
    destination.mkdir()
    records = []
    total = 0
    started = time.monotonic()
    for name in configuration["metadata"] + list(configuration["shards"]):
        url = f'https://huggingface.co/{configuration["model_id"]}/resolve/{configuration["model_revision"]}/{name}'
        response = requests.get(url, stream=True, timeout=45)
        response.raise_for_status()
        digest = hashlib.sha256()
        with (destination / name).open("xb") as output:
            for block in response.iter_content(1 << 20):
                total += len(block)
                if total > configuration["download_limit_bytes"]:
                    raise RuntimeError("Acquisition byte limit exceeded")
                output.write(block)
                digest.update(block)
        record = {"url": url, "path": f"model/{name}", "bytes": (destination / name).stat().st_size,
                  "sha256": digest.hexdigest()}
        if name in configuration["shards"]:
            assert record["bytes"] == configuration["shards"][name]["bytes"]
            assert record["sha256"] == configuration["shards"][name]["sha256"]
        records.append(record)
        (directory / "acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    print(json.dumps({"bytes": total, "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
