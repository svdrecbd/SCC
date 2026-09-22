"""Acquire the pinned public proof model and an isolated Lean release."""
import hashlib
import json
import sys
import time
from pathlib import Path
import requests


def main(root):
    config = json.loads((root / "config.json").read_text())
    files = [(config["lean_download_url"], "downloads/lean.tar.zst")]
    for name in ("README.md", "config.json", "generation_config.json", "model.safetensors",
                 "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json"):
        files.append((f'https://huggingface.co/{config["model_id"]}/resolve/'
                      f'{config["model_revision"]}/{name}', f"model/{name}"))
    for name in ("README.md", "LICENSE"):
        files.append((f'https://raw.githubusercontent.com/lean-dojo/ReProver/'
                      f'{config["source_revision"]}/{name}', f"upstream/{name}"))
    records = []
    total = 0
    started = time.monotonic()
    for url, relative in files:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        digest = hashlib.sha256()
        with destination.open("xb") as stream:
            for block in response.iter_content(1 << 20):
                total += len(block)
                if total > config["download_limit_bytes"]:
                    raise RuntimeError("Acquisition byte limit exceeded")
                stream.write(block)
                digest.update(block)
        if relative == "model/model.safetensors":
            assert digest.hexdigest() == config["model_sha256"]
        if relative == "downloads/lean.tar.zst":
            assert destination.stat().st_size == config["lean_download_bytes"]
        record = {"url": url, "path": relative, "bytes": destination.stat().st_size,
                  "sha256": digest.hexdigest()}
        records.append(record)
        (root / "acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({"path": relative, "bytes": record["bytes"]}), flush=True)
    print(json.dumps({"total_bytes": total, "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
