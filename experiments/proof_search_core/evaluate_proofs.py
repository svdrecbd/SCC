"""Compare fixed public tactics and an unchanged neural policy on Lean goals.

Development calibration only. Every accepted tactic is replayed in an environment
that does not contain the placeholder declaration, and its axioms are inspected.
"""
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


class ProofEvaluator:
    def __init__(self, configuration, output_directory, session_index=0):
        self.configuration = configuration
        self.output_directory = output_directory
        self.session_index = session_index
        session_directory = output_directory / f"evaluator_{session_index:03d}"
        session_directory.mkdir()
        environment = dict(os.environ)
        environment["PATH"] = configuration["lean_bin"] + ":" + environment["PATH"]
        self.transcript = (session_directory / "transcript.jsonl").open("x")
        self.process = subprocess.Popen(
            ["lake", "env", configuration["repl"]],
            cwd=configuration["mathlib"], env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(session_directory / "stderr.txt").open("x"),
            bufsize=0, start_new_session=True,
        )
        self.environment = self.request({"cmd": "import Mathlib\nset_option maxHeartbeats 100000\n"}, 30)["env"]

    def request(self, command, timeout=10):
        started = time.monotonic()
        self.process.stdin.write((json.dumps(command) + "\n\n").encode())
        self.process.stdin.flush()
        response = b""
        self.transcript.write(json.dumps({"request": command}) + "\n")
        self.transcript.flush()
        while True:
            remaining = timeout - (time.monotonic() - started)
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, remaining))
            if not ready:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
                raise TimeoutError("Proof evaluator response exceeded wall limit")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block and self.process.poll() is not None:
                raise RuntimeError("Proof evaluator terminated without a response")
            response += block
            try:
                result = json.loads(response)
                break
            except json.JSONDecodeError:
                continue
        self.transcript.write(json.dumps({"request": command, "response": result,
                                         "seconds": time.monotonic() - started}) + "\n")
        self.transcript.flush()
        return result

    def attempt(self, task, tactic):
        started = time.monotonic()
        if re.search(r"\b(sorry|admit|axiom|unsafe|run_tac|native_decide|implemented_by|extern|set_option|elab|macro)\b|#|\x00", tactic):
            return {"accepted": False, "reason": "unsupported proof execution", "seconds": 0}
        name = "verifiedDevelopmentProof"
        declaration = task.get("definitions", "") + "\nset_option maxHeartbeats 10000 in\n" + f"theorem {name} {task['statement']} := by\n"
        declaration += "\n".join("  " + line for line in tactic.splitlines())
        declaration += f"\n#print axioms {name}\n"
        response = self.request({"cmd": declaration, "env": self.environment})
        messages = response.get("messages", [])
        errors = [message for message in messages if message.get("severity") == "error"]
        axioms = [message.get("data", "") for message in messages if "depends on axioms" in message.get("data", "") or "does not depend on any axioms" in message.get("data", "")]
        allowed_axioms = {"propext", "Classical.choice", "Quot.sound"}
        listed_axioms = set()
        for entry in axioms:
            match = re.search(r"\[(.*?)\]", entry, re.DOTALL)
            if match:
                listed_axioms.update(name.strip() for name in match.group(1).split(",") if name.strip())
        accepted = not errors and not response.get("sorries") and bool(axioms) and listed_axioms <= allowed_axioms
        return {"accepted": accepted, "seconds": time.monotonic() - started,
                "axioms": axioms, "errors": errors}

    def state(self, task):
        response = self.request({"cmd": task.get("definitions", "") + "\n" +
            f"theorem developmentPlaceholder {task['statement']} := by sorry", "env": self.environment})
        assert not any(message.get("severity") == "error" for message in response.get("messages", [])), response
        return response["sorries"][0]["goal"]

    def bounded_attempt(self, task, tactic):
        started = time.monotonic()
        try:
            return self.attempt(task, tactic)
        except TimeoutError:
            elapsed = time.monotonic() - started
            self.transcript.write(json.dumps({"timeout": True, "seconds": elapsed}) + "\n")
            self.transcript.close()
            restarted = time.monotonic()
            self.__init__(self.configuration, self.output_directory, self.session_index + 1)
            return {"accepted": False, "reason": "response timeout", "seconds": elapsed,
                    "restart_seconds": time.monotonic() - restarted}

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.transcript.close()


def main(output_directory):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    configuration = json.loads((output_directory / "evaluation_config.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluator = ProofEvaluator(configuration, output_directory)
    results = []
    started = time.monotonic()
    for task in configuration["tasks"]:
        record = {"task": task["name"], "state": evaluator.state(task), "public": [], "neural": []}
        for tactic in task.get("public_templates", []) + configuration["public_tactics"]:
            observation = evaluator.bounded_attempt(task, tactic)
            record["public"].append({"tactic": tactic, **observation})
            if observation["accepted"]:
                break
        results.append(record)
        (output_directory / "public_results.json").write_text(json.dumps(results, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(configuration["model"], local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(configuration["model"], local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True).eval()
    for task, record in zip(configuration["tasks"], results):
        inference_started = time.monotonic()
        encoded = tokenizer(record["state"], return_tensors="pt")
        with torch.inference_mode():
            outputs = model.generate(**encoded, max_new_tokens=configuration["tokens"],
                num_beams=configuration["beams"], num_return_sequences=configuration["beams"], do_sample=False)
        record["inference_seconds"] = time.monotonic() - inference_started
        for output in outputs:
            tactic = tokenizer.decode(output, skip_special_tokens=True)
            record["neural"].append({"tactic": tactic, **evaluator.bounded_attempt(task, tactic)})
        (output_directory / "results.json").write_text(json.dumps({"classification": "development calibration",
            "training": False, "results": results, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
        print(json.dumps({"task": record["task"], "public_accepted": any(item["accepted"] for item in record["public"]),
            "neural_accepted": any(item["accepted"] for item in record["neural"]),
            "inference_seconds": record["inference_seconds"]}), flush=True)
    evaluator.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
