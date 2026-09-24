"""Matched unchanged-checkpoint and public path construction, LN-370."""

from pathlib import Path
import hashlib
import json
import math
import platform
import random
import resource
import sys
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from released_model import BiLevelCertificates
from create_graph import generate_er_graph, generate_ba_graph
from generate_grid import generate_exact_grid_graph
from graph_update import (assign_random_positions, compute_edge_length,
                          assign_node_features_directed, add_spectral_features_directed)
from decoding import decode_sampled_path_improved


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_path(graph, path, claimed_cost=None):
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        return False, None
    if path[0] != 0 or path[-1] != 1 or len(set(path)) != len(path):
        return False, None
    if any(not graph.has_edge(left, right) for left, right in zip(path, path[1:])):
        return False, None
    cost = math.fsum(graph[left][right]["weight"] for left, right in zip(path, path[1:]))
    if not math.isfinite(cost) or (claimed_cost is not None and abs(cost - claimed_cost) > 1e-7):
        return False, None
    return True, cost


def verify_controls():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 2, 0.0), (2, 1, -1.0), (2, 3, -2.0), (3, 2, -2.0)])
    assert verify_path(graph, [0, 2, 1], -1.0)[0]
    for path in (None, [], [0, 1], [0, 2], [2, 1], [0, 2, 3, 2, 1]):
        assert not verify_path(graph, path)[0]
    assert not verify_path(graph, [0, 2, 1], -2.0)[0]
    fixture = nx.DiGraph()
    fixture.add_weighted_edges_from([(0, 2, 0.0), (2, 1, -1.0)])
    path = decode_sampled_path_improved(fixture, list(fixture.edges()), torch.ones(2),
                                       0, 1, num_samples=3, seed=37037)
    assert verify_path(fixture, path, -1.0)[0]
    return 9


def create_problem(configuration):
    seed, count = configuration["seed"], configuration["nodes"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kind = configuration["graph_type"]
    if kind == "ER":
        graph = generate_er_graph(count, float(np.random.uniform(0.1, 0.5)), True)
    elif kind == "BA":
        ratio = np.random.uniform(0.05, 0.25)
        graph = generate_ba_graph(count, max(1, min(count - 3, int(ratio * count))), True)
    elif kind == "GRID":
        graph, _, _ = generate_exact_grid_graph(count)
    else:
        raise ValueError(kind)
    assign_random_positions(graph)
    compute_edge_length(graph, [-1.0, 1.0], 0, 1)
    assign_node_features_directed(graph, 0, 1)
    add_spectral_features_directed(graph, k=4)
    nodes = list(graph.nodes())
    assert nodes == list(range(count))
    edges = list(graph.edges())
    features = torch.tensor([graph.nodes[node]["x"] + graph.nodes[node]["spec"] for node in nodes], dtype=torch.float32)
    indices = torch.tensor(edges, dtype=torch.int64).T.contiguous()
    costs = torch.tensor([graph[left][right]["weight"] for left, right in edges], dtype=torch.float32)
    # The actual solver instance uses the released model's float32 cost precision.
    for (left, right), cost in zip(edges, costs.tolist()):
        graph[left][right]["weight"] = cost
    assert features.shape == (count, 19) and torch.isfinite(features).all()
    data = Data(x=features, edge_index=indices, edge_attr=costs[:, None], source=0, sink=1)
    assert set(data.keys()) == {"x", "edge_index", "edge_attr", "source", "sink"}
    return graph, data


def greedy_plan(graph):
    completed = []
    # Every released graph has source -> vertex -> sink with zero boundary costs.
    for first in graph.successors(0):
        path = [0, first]
        visited = set(path)
        while path[-1] != 1:
            available = [node for node in graph.successors(path[-1]) if node not in visited]
            if not available:
                break
            following = min(available, key=lambda node: (graph[path[-1]][node]["weight"], node))
            path.append(following)
            visited.add(following)
        valid, cost = verify_path(graph, path)
        if valid:
            completed.append((cost, path))
    return min(completed, key=lambda item: item[0])[1] if completed else []


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    resource.setrlimit(resource.RLIMIT_FSIZE, (10485760, 10485760))
    torch.set_num_threads(1)
    for name, expected in configuration["source_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    checkpoint_path = Path(configuration["checkpoint"])
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert checkpoint_digest == configuration["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arguments = checkpoint["args"]
    model = BiLevelCertificates(checkpoint["node_dim"], hidden_dim=arguments["hidden_dim"],
                                K=arguments["K"], tau=0.2, gamma=arguments["gamma"], d_max=arguments["d_max"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    original_digest = state_digest(model)
    controls = verify_controls()
    preparation_start = time.monotonic()
    graph, data = create_problem(configuration)
    preparation_seconds = time.monotonic() - preparation_start
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        guidance_start = time.monotonic()
        initial_potentials = model.gnn(batch.x, batch.edge_index, batch.edge_attr[:, 0])
        guidance_seconds = time.monotonic() - guidance_start
        methods = {}
        for name, potentials in (("learned", initial_potentials), ("zero_potential", torch.zeros_like(initial_potentials))):
            recovery_start = time.monotonic()
            recovered = model._per_graph(potentials, data.edge_index, data.edge_attr[:, 0],
                                         configuration["nodes"], 0, 1, torch.device("cpu"))
            probabilities = recovered["P"][data.edge_index[0], data.edge_index[1]].clone()
            assert torch.isfinite(probabilities).all()
            methods[name] = (probabilities, time.monotonic() - recovery_start)
        sampler_start = time.monotonic()
        cost_probabilities = torch.zeros(data.edge_index.shape[1])
        for node in range(configuration["nodes"]):
            selection = data.edge_index[0] == node
            if selection.any():
                cost_probabilities[selection] = torch.softmax(-data.edge_attr[selection, 0] / 0.2, dim=0)
        methods["cost_sampler"] = (cost_probabilities, time.monotonic() - sampler_start)
    results = {}
    for name, (probabilities, recovery_seconds) in methods.items():
        decode_start = time.monotonic()
        path = decode_sampled_path_improved(graph, list(graph.edges()), probabilities, 0, 1,
            num_samples=configuration["samples"], seed=configuration["decode_seed"], tau=1.0)
        decode_seconds = time.monotonic() - decode_start
        verification_start = time.monotonic()
        valid, cost = verify_path(graph, path)
        verification_seconds = time.monotonic() - verification_start
        results[name] = {"valid": valid, "cost": cost,
                         "path": [int(node) for node in path] if valid else None,
                         "guidance_seconds": guidance_seconds if name == "learned" else 0,
                         "recovery_seconds": recovery_seconds, "decode_seconds": decode_seconds,
                         "verification_seconds": verification_seconds}
    greedy_start = time.monotonic()
    path = greedy_plan(graph)
    greedy_seconds = time.monotonic() - greedy_start
    valid, cost = verify_path(graph, path)
    results["greedy_all_starts"] = {"valid": valid, "cost": cost, "path": path if valid else None,
                                     "construction_seconds": greedy_seconds}
    assert state_digest(model) == original_digest
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_digest
    graph_record = {"nodes": configuration["nodes"], "edges": [[left, right, values["weight"]]
        for left, right, values in graph.edges(data=True)], "features": data.x.tolist()}
    (directory / "problem.json").write_text(json.dumps(graph_record) + "\n")
    result = {"graph_type": configuration["graph_type"], "seed": configuration["seed"],
              "checkpoint_sha256": checkpoint_digest, "state_sha256": original_digest,
              "unchanged": True, "controls": controls, "feature_seconds": preparation_seconds,
              "methods": results, "total_seconds": time.monotonic() - started,
              "optimum": "unknown", "classification": "development screen"}
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "runtime.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "networkx": nx.__version__}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
