"""Generate one unchanged-model proof per fixed program task; no model updates."""
import json
import sys
import time
from pathlib import Path
import torch
import torch.backends.python_native as python_native
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def main(directory):
    configuration = json.loads((directory / "interface_config.json").read_text())
    cases = json.loads((directory / "program_tasks.json").read_text())
    model_directory = Path(configuration["model"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    python_native.triton.enabled = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    settings = json.loads((model_directory / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_directory / "tokenizer.json"),
        bos_token=settings["bos_token"]["content"], eos_token=settings["eos_token"]["content"],
        pad_token=settings["pad_token"]["content"], chat_template=settings["chat_template"],
        clean_up_tokenization_spaces=False)
    raw_tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    placement = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    placement.update({f"model.layers.{index}": 0 if index < 20 else 1 for index in range(30)})
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_directory, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
        device_map=placement, attn_implementation="eager").eval()
    assert {str(parameter.device) for parameter in model.parameters()} <= {"cuda:0", "cuda:1"}
    results = []
    loaded = time.monotonic()
    prior_responses = {}
    prior_checks = {}
    if cases.get("use_checker_feedback"):
        prior_responses = {record["task"]: record for record in json.loads((directory / "parent_generation.json").read_text())["results"]}
        prior_checks = {record["task"]: record for record in json.loads((directory / "parent_verification.json").read_text())["records"]}
    for task in cases["tasks"]:
        prompt = "Complete the fixed Lean 4 theorem below. Return one lean4 code block containing ONLY the tactic proof that belongs after 'by'. Do not repeat the theorem declaration or definitions. You may introduce local helper facts with 'have'. Do not use sorry, admit, added axioms, native_decide or external tools.\n\n"
        prompt += "```lean4\nimport Mathlib\n" + task["definitions"]
        prompt += f"\ntheorem requestedProof {task['statement']} := by\n  sorry\n```"
        messages = [{"role": "user", "content": prompt}]
        if cases.get("use_checker_feedback"):
            feedback = json.dumps(prior_checks[task["name"]].get("errors", prior_checks[task["name"]].get("reason", "No complete proof accepted")), ensure_ascii=False)
            messages += [{"role": "assistant", "content": prior_responses[task["name"]]["text"]},
                {"role": "user", "content": "The independent Lean checker rejected this proof of the original fixed statement. Its diagnostic follows:\n" + feedback +
                 "\nProvide a detailed proof plan before the corrected Lean 4 code. Explain which facts the induction hypothesis must establish. Consider a stronger invariant or a local helper lemma if the current hypothesis is insufficient. Do not change the statement or add assumptions. Return the corrected tactic proof in the final lean4 code block."}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        assert encoded["input_ids"][0].tolist() == raw_tokenizer.encode(rendered, add_special_tokens=False).ids
        encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
        generation_started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=cases["maximum_new_tokens"],
                                    do_sample=False, max_time=120)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
        record = {"task": task["name"], "prompt": prompt, "input_token_count": encoded["input_ids"].shape[1],
            "generated_tokens": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "seconds": time.monotonic() - generation_started,
            "ended_with_eos": bool(tokens and tokens[-1] == tokenizer.eos_token_id)}
        results.append(record)
        (directory / "results.json").write_text(json.dumps({"classification": cases["classification"],
            "training": False, "load_seconds": loaded - started, "results": results,
            "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(device) for device in range(2)],
            "seconds": time.monotonic() - started}, indent=2) + "\n")
        print(json.dumps({"task": task["name"], "tokens": len(tokens), "seconds": record["seconds"],
                          "ended_with_eos": record["ended_with_eos"]}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
