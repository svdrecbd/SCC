"""Audit the saved path screen and certify its public threshold-reader floor."""

from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import time

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from assess_elementary_path import state_digest, verify_path
from released_model import BiLevelCertificates
from decoding import decode_sampled_path_improved


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    results = []
    accuracy_floors = []
    retained_fractions = []
    for case in configuration["cases"]:
        stage = Path(case["directory"])
        for name, expected in case["sha256"].items():
            assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == expected
        problem = json.loads((stage / "problem.json").read_text())
        prior = json.loads((stage / "results.json").read_text())
        arguments = json.loads((stage / "config.json").read_text())
        checkpoint_path = Path(arguments["checkpoint"])
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == arguments["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        options = checkpoint["args"]
        model = BiLevelCertificates(19, hidden_dim=64, K=options["K"], tau=0.2,
                                    gamma=options["gamma"], d_max=options["d_max"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        original_digest = state_digest(model)
        assert original_digest == prior["state_sha256"]
        indices = torch.tensor([[left, right] for left, right, _ in problem["edges"]]).T.contiguous()
        costs = torch.tensor([cost for _, _, cost in problem["edges"]], dtype=torch.float32)
        data = Data(x=torch.tensor(problem["features"], dtype=torch.float32),
                    edge_index=indices, edge_attr=costs[:, None], source=0, sink=1)
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            native = model(batch)[0]
            potentials = model.gnn(data.x, indices, costs)
            staged = model._per_graph(potentials, indices, costs, problem["nodes"], 0, 1, torch.device("cpu"))
            public = model._per_graph(torch.zeros_like(potentials), indices, costs,
                                      problem["nodes"], 0, 1, torch.device("cpu"))
            constant = model._per_graph(torch.full_like(potentials, -options["d_max"]), indices, costs,
                                        problem["nodes"], 0, 1, torch.device("cpu"))
            implementation_error = float((native["P"] - staged["P"]).abs().max())
            assert implementation_error <= 1e-6
        assert state_digest(model) == original_digest
        graph = nx.DiGraph()
        graph.add_nodes_from(range(problem["nodes"]))
        graph.add_weighted_edges_from(problem["edges"])
        constant_probabilities = constant["P"][indices[0], indices[1]]
        constant_path = decode_sampled_path_improved(graph, list(graph.edges()), constant_probabilities, 0, 1,
            num_samples=arguments["samples"], seed=arguments["decode_seed"], tau=1.0)
        constant_valid, constant_cost = verify_path(graph, constant_path)
        assert constant_valid
        constant_identical = bool(torch.equal(potentials, torch.full_like(potentials, -options["d_max"])))
        if constant_identical:
            assert torch.equal(staged["P"], constant["P"])
            assert list(constant_path) == prior["methods"]["learned"]["path"]
        prior["methods"]["constant_potential"] = {"valid": True, "cost": constant_cost, "path": constant_path}
        exact_edges = {(left, right): Fraction.from_float(cost) for left, right, cost in problem["edges"]}
        upper = sum(max([Fraction(0)] + [-exact_edges[(node, following)]
                    for following in graph.successors(node)]) for node in graph.nodes())
        values = {}
        for name, result in prior["methods"].items():
            assert result["valid"] and verify_path(graph, result["path"], result["cost"])[0]
            path = result["path"]
            values[name] = -sum(exact_edges[(left, right)] for left, right in zip(path, path[1:]))
            assert 0 < values[name] <= upper
        best_method = max((name for name in values if name != "learned"), key=values.get)
        public_value = values[best_method]
        floor = Fraction(1, 2) + public_value / (2 * upper)
        accuracy_floors.append(floor)
        retention = min(Fraction(1), public_value / values["learned"])
        retained_fractions.append(retention)
        results.append({"graph_type": prior["graph_type"], "seed": prior["seed"],
                        "full_forward_transition_error": implementation_error,
                        "initial_potential_minimum": float(potentials.min()),
                        "initial_potential_maximum": float(potentials.max()),
                        "initial_potential_standard_deviation": float(potentials.std(unbiased=False)),
                        "saturated_potentials": int((potentials.abs() >= options["d_max"] - 1e-6).sum()),
                        "learned_zero_transition_difference": float((staged["P"] - public["P"]).abs().max()),
                        "constant_potential_exact_replacement": constant_identical,
                        "constant_potential_cost": constant_cost,
                        "constant_potential_path": [int(node) for node in constant_path],
                        "learned_cost": prior["methods"]["learned"]["cost"],
                        "best_public_method": best_method, "best_public_cost": -float(public_value),
                        "public_matches_or_exceeds": public_value >= values["learned"],
                        "public_retained_fraction": float(retention), "objective_upper_bound": str(upper),
                        "protected_accuracy_lower_bound": str(floor),
                        "protected_accuracy_lower_bound_decimal": float(floor)})
    average_floor = sum(accuracy_floors) / len(accuracy_floors)
    average_retention = sum(retained_fractions) / len(retained_fractions)
    summary = {"cases": len(results), "public_matches_or_exceeds": sum(row["public_matches_or_exceeds"] for row in results),
               "mean_capped_retained_fraction": float(average_retention),
               "mean_protected_accuracy_lower_bound": str(average_floor),
               "mean_protected_accuracy_lower_bound_decimal": float(average_floor),
               "seconds": time.monotonic() - started,
               "scope": "exact lower bound for paired thresholds on the six saved base graphs, not observed labels"}
    (directory / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n")
    print(json.dumps({"summary": summary, "cases": results}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
