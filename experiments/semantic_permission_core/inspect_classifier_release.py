"""Acquire bounded, pinned public model source without executing it."""

from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    receipts = []
    total_bytes = 0
    def retrieve(url, filename):
        nonlocal total_bytes
        if len(receipts) >= settings["maximum_requests"]:
            raise RuntimeError("request budget exhausted")
        remaining = settings["maximum_bytes"]-total_bytes
        if remaining <= 0:
            raise RuntimeError("byte budget exhausted")
        request = urllib.request.Request(url, headers={"User-Agent": "SCC-source-inspection"})
        with urllib.request.urlopen(request, timeout=settings["request_timeout_seconds"]) as response:
            body = response.read(remaining)
            record = {"url": url, "resolved_url": response.url, "status": response.status,
                      "filename": filename, "bytes": len(body),
                      "sha256": hashlib.sha256(body).hexdigest(),
                      "possibly_truncated": len(body) == remaining}
        total_bytes += len(body)
        receipts.append(record)
        (directory / filename).write_bytes(body)
        (directory / "request_receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
        if record["possibly_truncated"]:
            raise RuntimeError("response reaches remaining byte budget")
        return body
    repository = settings["repository"]
    commit = json.loads(retrieve("https://api.github.com/repos/"+repository+"/commits/"+settings["reference"], "commit.json"))
    revision = commit["sha"]
    tree = json.loads(retrieve("https://api.github.com/repos/"+repository+"/git/trees/"+revision+"?recursive=1", "tree.json"))
    assert not tree.get("truncated", False)
    paths = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    selected = list(settings["required_paths"])
    matching_sources = sorted(path for path in paths if path.endswith(".py")
                              and any(fragment in path for fragment in settings.get("source_name_fragments", [])))
    selected.extend(path for path in matching_sources if path not in selected)
    selected = selected[:settings["maximum_requests"]-2]
    notebooks = sorted(path for path in paths if path.startswith("colab/") and path.endswith(".ipynb"))
    selected.extend(notebooks[:settings["maximum_requests"]-2-len(selected)])
    sources = []
    for index, path in enumerate(selected):
        assert path in paths, path
        filename = "source_"+str(index).zfill(2)+"_"+Path(path).name
        body = retrieve("https://raw.githubusercontent.com/"+repository+"/"+revision+"/"+path, filename)
        object_digest = hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest()
        assert object_digest == paths[path]["sha"], path
        sources.append({"path": path, "local_filename": filename, "git_blob": object_digest})
    result = {"status": "complete", "repository": repository, "revision": revision,
              "sources": sources, "requests": len(receipts), "total_bytes": total_bytes,
              "upstream_code_executed": False, "weights_downloaded": False,
              "neural_training": False, "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
