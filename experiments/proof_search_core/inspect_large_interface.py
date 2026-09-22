"""Qualify split-device, unchanged whole-proof-model inference on Charon."""
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
import torch
import torch.backends.python_native as python_native
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer


def main(directory):
    configuration = json.loads((directory / "interface_config.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    python_native.triton.enabled = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    model_directory = Path(configuration["model"])
    tokenizer_configuration = json.loads((model_directory / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_directory / "tokenizer.json"),
        bos_token=tokenizer_configuration["bos_token"]["content"],
        eos_token=tokenizer_configuration["eos_token"]["content"],
        pad_token=tokenizer_configuration["pad_token"]["content"],
        chat_template=tokenizer_configuration["chat_template"], clean_up_tokenization_spaces=False)
    raw_tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    sample = "theorem validation (n : ℕ) : n + 0 = n := by\n  rfl\n∀ x, x = x"
    reference_tokens = raw_tokenizer.encode(sample, add_special_tokens=False).ids
    assert tokenizer.encode(sample, add_special_tokens=False) == reference_tokens
    assert tokenizer.decode(reference_tokens, skip_special_tokens=False) == sample
    placement = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    placement.update({f"model.layers.{index}": 0 if index < 20 else 1 for index in range(30)})
    model = AutoModelForCausalLM.from_pretrained(configuration["model"],
        trust_remote_code=False, local_files_only=True, use_safetensors=True,
        dtype=torch.float16, attn_implementation="eager", device_map=placement).eval()
    devices = {str(parameter.device) for parameter in model.parameters()}
    (directory / "placement.json").write_text(json.dumps({"device_map": model.hf_device_map,
        "parameter_devices": sorted(devices), "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()})}, indent=2) + "\n")
    assert devices <= {"cuda:0", "cuda:1"} and devices, devices
    assert sum(parameter.numel() for parameter in model.parameters()) == 6910365696
    messages = [{"role": "user", "content": configuration["prompt"]}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert encoded["input_ids"][0].tolist() == raw_tokenizer.encode(rendered_prompt, add_special_tokens=False).ids
    encoded = {name: value.to(model.get_input_embeddings().weight.device) for name, value in encoded.items()}
    loaded = time.monotonic()
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
        assert torch.isfinite(logits).all().item()
        output = model.generate(**encoded, max_new_tokens=configuration["tokens"], do_sample=False)
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - loaded
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    result = {"classification": "GPU interface check only", "training": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "float16", "attention": "eager", "automatic_triton": python_native.triton.enabled,
        "tokenizer_matches_serialized_backend": True, "unicode_round_trip": True,
        "device_map": model.hf_device_map, "finite_prefill_logits": True,
        "load_seconds": loaded - started, "inference_seconds": elapsed,
        "generated_token_ids": tokens, "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
        "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(device) for device in range(torch.cuda.device_count())],
        "gpu_peak_reserved_bytes": [torch.cuda.max_memory_reserved(device) for device in range(torch.cuda.device_count())],
        "cpu_peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version,
        "platform": platform.platform(), "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_names": [torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())]}
    (directory / "interface.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
