"""Acquire bounded public release metadata without executing acquired content."""

from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import urllib.error
import urllib.request


def main(directory):
    configuration = json.loads((directory / "config.json").read_text())
    assert len(configuration["requests"]) <= configuration["maximum_requests"]
    started = time.monotonic()
    total_bytes = 0
    receipts = []
    for index, request in enumerate(configuration["requests"]):
        if total_bytes >= configuration["maximum_bytes"]:
            break
        remaining = configuration["maximum_seconds"] - (time.monotonic() - started)
        if remaining <= 1:
            receipts.append({"url": request["url"], "status": "not requested: time limit"})
            break
        receipt = {"url": request["url"], "output": request["output"]}
        path = directory / request["output"]
        assert path.parent == directory and not path.exists()
        try:
            query = urllib.request.Request(request["url"], headers={"User-Agent": "SCC-Research-Release-Inspection/1.0"})
            with urllib.request.urlopen(query, timeout=min(8, remaining)) as response:
                allowance = configuration["maximum_bytes"] - total_bytes
                data = response.read(allowance)
                total_bytes += len(data)
                path.write_bytes(data)
                receipt.update({"status": response.status, "bytes": len(data),
                                "possibly_truncated": len(data) == allowance,
                                "content_type": response.headers.get("Content-Type"),
                                "final_url": response.url,
                                "sha256": hashlib.sha256(data).hexdigest()})
        except (urllib.error.URLError, TimeoutError, AssertionError) as error:
            receipt["status"] = "unavailable"
            receipt["error"] = str(error)
        receipts.append(receipt)
        (directory / "receipts.json").write_text(json.dumps({"requests": receipts, "bytes": total_bytes}, indent=2) + "\n")
    result = {"requests": receipts, "bytes": total_bytes, "seconds": time.monotonic() - started}
    (directory / "receipts.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "runtime.json").write_text(json.dumps({"platform": platform.platform(), "python": sys.version}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
