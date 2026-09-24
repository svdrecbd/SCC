"""Download bounded published resources, retaining partial files and receipts."""

from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request


def main(directory):
    started = time.monotonic()
    settings = json.loads((directory / "config.json").read_text())
    receipts = []
    total_bytes = 0
    for resource in settings["resources"]:
        record = dict(resource, status="started", bytes=0)
        receipts.append(record)
        (directory / "receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
        digest = hashlib.sha256()
        path = directory / resource["filename"]
        try:
            request = urllib.request.Request(resource["url"], headers={"User-Agent": "SCC-resource-acquisition"})
            with urllib.request.urlopen(request, timeout=settings["request_timeout_seconds"]) as response, path.open("xb") as output:
                record.update(resolved_url=response.url, http_status=response.status,
                              headers=dict(response.headers))
                while True:
                    if time.monotonic()-started > settings["maximum_seconds"]-5:
                        raise TimeoutError("stage wall-time budget reached")
                    remaining = settings["maximum_bytes"]-total_bytes
                    if remaining <= 0:
                        raise RuntimeError("aggregate byte budget reached")
                    block = response.read1(min(1048576, remaining))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    record["bytes"] += len(block)
                    total_bytes += len(block)
                    (directory / "receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
            record["sha256"] = digest.hexdigest()
            if "sha256_prefix" in resource:
                assert record["sha256"].startswith(resource["sha256_prefix"])
            if "sha256" in resource:
                assert record["sha256"] == resource["sha256"]
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", error=repr(error), partial_sha256=digest.hexdigest())
            (directory / "receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
            raise
        (directory / "receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
    result = {"status": "complete", "bytes": total_bytes, "resources": receipts,
              "wall_seconds": time.monotonic()-started, "code_executed_from_resources": False,
              "neural_training": False}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "resources"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
