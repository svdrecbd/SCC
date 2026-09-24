"""Replay the path portfolio without learned weights, features, or model imports."""

from pathlib import Path
import hashlib
import json
import math
import sys
import time

import networkx as nx
import torch

from public_path_recovery import PublicPathRecovery
from decoding import decode_sampled_path_improved


def reject_checkpoint_access(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0]).lower()
        if ".pt" in Path(path).suffixes or path.endswith((".pth", ".ckpt")):
            raise RuntimeError("Checkpoint access is prohibited in the public replacement")


def verified_cost(graph, path):
    assert len(path) >= 2 and path[0] == 0 and path[-1] == 1
    assert len(set(path)) == len(path)
    assert all(graph.has_edge(left, right) for left, right in zip(path, path[1:]))
    result = math.fsum(graph[left][right]["weight"] for left, right in zip(path, path[1:]))
    assert math.isfinite(result)
    return result


def greedy_constructor(graph):
    completed = []
    for first in graph.successors(0):
        path, visited = [0, first], {0, first}
        while path[-1] != 1:
            available = [node for node in graph.successors(path[-1]) if node not in visited]
            if not available:
                break
            following = min(available, key=lambda node: (graph[path[-1]][node]["weight"], node))
            path.append(following)
            visited.add(following)
        if path[-1] == 1:
            completed.append((verified_cost(graph, path), path))
    return min(completed, key=lambda item: item[0])[1]


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    sys.addaudithook(reject_checkpoint_access)
    controls = 0
    try:
        open(directory / "prohibited_checkpoint.pt", "rb")
    except RuntimeError:
        controls += 1
    assert controls == 1
    recovery = PublicPathRecovery(K=15, tau=0.2, gamma=0.7)
    results = []
    for case in configuration["cases"]:
        data = (directory / case["input"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == case["sha256"]
        problem = json.loads(data)
        assert set(problem) == {"nodes", "edges"}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(problem["nodes"]))
        graph.add_weighted_edges_from(problem["edges"])
        indices = torch.tensor([[left, right] for left, right, _ in problem["edges"]]).T.contiguous()
        costs = torch.tensor([cost for _, _, cost in problem["edges"]], dtype=torch.float32)
        case_started = time.monotonic()
        probabilities = {}
        with torch.no_grad():
            for method, constant in (("zero_potential", 0), ("constant_potential", -2)):
                potentials = torch.full((problem["nodes"],), constant, dtype=torch.float32)
                recovered = recovery._per_graph(potentials, indices, costs, problem["nodes"], 0, 1, torch.device("cpu"))
                probabilities[method] = recovered["P"][indices[0], indices[1]]
            cost_probabilities = torch.zeros(len(problem["edges"]))
            for node in range(problem["nodes"]):
                selected = indices[0] == node
                if selected.any():
                    cost_probabilities[selected] = torch.softmax(-costs[selected] / 0.2, dim=0)
            probabilities["cost_sampler"] = cost_probabilities
        paths = {}
        for method, probability in probabilities.items():
            path = decode_sampled_path_improved(graph, list(graph.edges()), probability, 0, 1,
                                                num_samples=300, seed=37037, tau=1.0)
            paths[method] = [int(node) for node in path]
        paths["greedy_all_starts"] = greedy_constructor(graph)
        values = {method: verified_cost(graph, path) for method, path in paths.items()}
        selected = min(values, key=values.get)
        results.append({"input": case["input"], "methods": values, "selected_method": selected,
                        "paths": paths, "selected_cost": values[selected], "selected_path": paths[selected],
                        "seconds": time.monotonic() - case_started})
    assert "released_model" not in sys.modules and "torch_geometric" not in sys.modules
    summary = {"status": "constructed", "cases": len(results), "returned_paths": 4 * len(results),
               "checkpoint_access_rejection_controls": controls, "learned_model_imported": False,
               "feature_inputs": False, "seconds": time.monotonic() - started}
    (directory / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "cases": results}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
