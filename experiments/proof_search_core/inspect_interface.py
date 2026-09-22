"""Inspect a fixed public proof model without changing its weights."""
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main(root):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((root / "config.json").read_text())
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(root / "model", local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == config["model_parameters"]
    loaded = time.monotonic()
    state = "n : Nat\n⊢ n + 0 = n"
    encoded = tokenizer(state, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=48, num_beams=1, do_sample=False)
    tactic = tokenizer.decode(output[0], skip_special_tokens=True)
    result = {"classification": "interface check only; no useful-advantage evidence",
              "parameters": parameter_count, "state": state, "tactic": tactic,
              "load_seconds": loaded - started,
              "inference_seconds": time.monotonic() - loaded,
              "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "affinity": sorted(os.sched_getaffinity(0)), "training": False}
    (root / "interface.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "machine.json").write_text(json.dumps({"hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "transformers": transformers.__version__}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
