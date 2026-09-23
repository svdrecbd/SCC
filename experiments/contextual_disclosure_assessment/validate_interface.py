"""Validate a preserved frozen model with local context and no network access."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def prohibit_connection(*arguments, **keywords):
    raise RuntimeError("Network access is excluded from model validation")


socket.socket.connect = prohibit_connection
socket.create_connection = prohibit_connection


def parameter_digest(models):
    digest = hashlib.sha256()
    count = 0
    for model in models:
        for name, parameter in model.state_dict().items():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        count += sum(parameter.numel() for parameter in model.parameters())
    return digest.hexdigest(), count


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    import numpy as np
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = Path(configuration["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == configuration["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload_description = {key: type(value).__name__ for key, value in payload.items()}
    generator = np.random.default_rng(configuration["seed"])
    context = generator.normal(size=(48, 4))
    labels = np.tile(np.arange(3), 16)
    context[:, 0] += labels
    queries = generator.normal(size=(8, 4))
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, model_path=checkpoint, device="cpu", n_estimators=1,
        random_state=configuration["seed"], n_preprocessing_jobs=1,
        inference_precision=torch.float32, tuning_config=None, show_progress_bar=False)
    fit_started = time.monotonic()
    model.fit(context, labels)
    fit_seconds = time.monotonic() - fit_started
    initial_digest, parameter_count = parameter_digest(model.models_)
    prediction_started = time.monotonic()
    probability = model.predict_proba(queries)
    prediction_seconds = time.monotonic() - prediction_started
    repeated = model.predict_proba(queries)
    final_digest, _ = parameter_digest(model.models_)
    assert initial_digest == final_digest
    assert np.allclose(probability, repeated, atol=1e-7)
    assert probability.shape == (8, 3)
    assert np.all(np.isfinite(probability)) and np.all(probability >= 0)
    assert np.allclose(probability.sum(axis=1), 1, atol=1e-6)
    risks = 1 - np.cumsum(probability, axis=1)[:, :-1]
    recovered = -np.diff(np.column_stack((np.ones(8), risks, np.zeros(8))), axis=1)
    assert np.allclose(recovered, probability, atol=1e-6)
    result = {"parameter_count": parameter_count, "state_sha256": initial_digest,
              "payload_types": payload_description, "classes": model.classes_.tolist(),
              "probability": probability.tolist(), "fit_seconds": fit_seconds,
              "prediction_seconds": prediction_seconds, "elapsed_seconds": time.monotonic() - started,
              "repeated_prediction_maximum_difference": float(np.max(abs(probability-repeated))),
              "parameter_state_unchanged": True, "training": False}
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "probability"}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
