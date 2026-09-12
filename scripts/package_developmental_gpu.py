"""Freeze a minimal GPU context; never include sealed test blocks or credentials."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from scc.provenance import atomic_json, file_digest, snapshot_sources


def package(output, parents=None):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = output / "context"
    snapshot_sources(context)
    for name in ("AGENTS.md", "WORKING_STANDARDS.md", "MECHANISM_TARGET.md", "protocols/DEVELOPMENTAL_COUPLING_V1.md", "protocols/DEVELOPMENTAL_COUPLING_V2.md", "protocols/DEVELOPMENTAL_COUPLING_V3.md", "protocols/SCC_DIAGNOSTICS_V1.md", "protocols/SCC_FULL_GRADIENT_PILOT_V1.md", "tests/test_developmental.py", "tests/test_causal_interventions.py", "tests/test_gradient_diagnostics.py", "tests/test_differentiable_modify.py", "scripts/package_developmental_gpu.py"):
        target = context / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    (context / "data").mkdir()
    source = root / "artifacts/retrieval-recovery/byte-prepared"
    for name in ("manifest.json", "train.tokens.bin", "train.labels.bin", "train.groups.bin", "validation.tokens.bin", "validation.labels.bin", "validation.groups.bin"):
        shutil.copyfile(source / name, context / "data" / name)
    if parents:
        for arm in ("rule_only", "early", "late"):
            target = context / "parents" / arm
            target.mkdir(parents=True)
            for name in ("step-00018000.pt", "result.json", "contract.json"):
                shutil.copyfile(Path(parents) / arm / name, target / name)
        references = {}
        for key, name in (("validation", "untrained.json"), ("reordered", "untrained-reordered.json")):
            record = json.loads((Path(parents) / name).read_text())
            references[key] = {field: record[field] for field in ("tasks", "text")}
        references["provenance"] = {name: file_digest(Path(parents) / name)
                                     for name in ("untrained.json", "untrained-reordered.json")}
        atomic_json(context / "pilot-untrained.json", references)
    (context / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
        "RUN python -m pip install --no-cache-dir torch==2.14.0 numpy==2.5.3 tokenizers==0.23.2\n"
        "RUN python -c \"import sys,torch; assert sys.version_info[:2] == (3,13); assert torch.__version__.split('+')[0] == '2.14.0'\"\n"
        "WORKDIR /workspace\nCOPY . /workspace\n"
        "ENV PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
    )
    manifest = {str(p.relative_to(context)): file_digest(p) for p in sorted(context.rglob("*")) if p.is_file()}
    atomic_json(output / "context_manifest.json", manifest)
    archive = output / "context.tar.zst"
    with archive.open("wb") as target:
        process = subprocess.Popen(["zstd", "-q", "-T2", "-c"], stdin=subprocess.PIPE, stdout=target)
        with tarfile.open(fileobj=process.stdin, mode="w|") as bundle:
            for path in sorted(context.rglob("*")):
                if path.is_file():
                    bundle.add(path, arcname=str(path.relative_to(context)))
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Context compression failed")
    result = {"path": str(archive.resolve()), "size_bytes": archive.stat().st_size, "sha256": file_digest(archive),
              "sealed_test_data_uploaded": False}
    atomic_json(output / "archive.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--parents", help="Preserved developmental comparison directory")
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.parents)))
