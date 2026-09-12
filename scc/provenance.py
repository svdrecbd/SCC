"""Small, explicit artifact and source fingerprints."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_manifest():
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "scc").glob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")
              if (root / name).is_file()]
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def snapshot_sources(destination):
    root = Path(__file__).resolve().parent.parent
    destination = Path(destination)
    manifest = source_manifest()
    for relative in manifest:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
        if file_digest(target) != manifest[relative]:
            raise ValueError("Source changed during snapshot")
    atomic_json(destination / "source_manifest.json", manifest)
    return manifest
