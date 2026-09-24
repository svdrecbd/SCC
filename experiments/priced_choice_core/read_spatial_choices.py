"""Read only packed actions and recover complementary event probabilities."""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    forbidden = {"evaluation_data.npz", "public_parameters.npz", "queries.json"}
    def access_check(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            name = Path(arguments[0]).name
            if name in forbidden or name.startswith("prediction_") or name.endswith((".pt", ".safetensors", ".h5", ".tar")):
                raise PermissionError("Reader cannot access hidden assessment inputs")
    sys.addaudithook(access_check)
    try:
        (directory.parent / "policy01/evaluation_data.npz").read_bytes()
    except PermissionError:
        rejected = True
    else:
        raise AssertionError("evaluation access was not rejected")
    path = directory / "policy_actions.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == settings["actions_sha256"]
    packed = np.load(path, allow_pickle=False)["actions"]
    actions = np.unpackbits(packed, axis=-1, count=settings["reader_calls"], bitorder="little")
    estimates = 1 - actions.mean(axis=-1)
    recoded = 1 - actions
    restored_estimates = 1 - (1 - recoded).mean(axis=-1)
    assert np.array_equal(estimates, restored_estimates)
    np.save(directory / "recovered_probabilities.npy", estimates, allow_pickle=False)
    result = {"status": "complete", "actions_sha256": digest,
              "shape": list(estimates.shape), "actual_binary_action_count": int(actions.size),
              "forbidden_input_control_rejected": rejected, "inverted_action_recovery_exact": True,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
