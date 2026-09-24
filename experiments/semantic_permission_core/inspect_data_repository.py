"""Read public versioned dataset metadata and prepare a hash-pinned download."""

from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request


def main(directory):
    started = time.perf_counter()
    url = "https://huggingface.co/api/datasets/uoft-cs/cifar100?blobs=true"
    receipts = []
    def retrieve(address, name):
        request = urllib.request.Request(address, headers={"User-Agent": "SCC-data-inspection"})
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read(1048576)
            assert len(body) < 1048576
            receipts.append({"url": address, "resolved_url": response.url,
                             "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        (directory / name).write_bytes(body)
        (directory / "receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
        return body
    metadata = json.loads(retrieve(url, "repository.json"))
    revision = metadata["sha"]
    candidates = [record for record in metadata["siblings"]
                  if "/test-" in record["rfilename"] and record["rfilename"].endswith(".parquet")]
    assert len(candidates) == 1
    selected = candidates[0]
    assert selected["size"] < 64*1024**2
    retrieve("https://huggingface.co/datasets/uoft-cs/cifar100/raw/"+revision+"/README.md", "README.md")
    configuration = {"maximum_bytes": 64*1024**2, "maximum_seconds": 60, "request_timeout_seconds": 10,
                     "neural_training": False, "resources": [{
                         "url": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/"+revision+"/"+selected["rfilename"],
                         "filename": "cifar100_test.parquet", "sha256": selected["lfs"]["sha256"]}]}
    (directory / "download_config.json").write_text(json.dumps(configuration, indent=2)+"\n")
    result = {"status": "complete", "revision": revision, "selected_file": selected,
              "requests": len(receipts), "bytes": sum(record["bytes"] for record in receipts),
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
