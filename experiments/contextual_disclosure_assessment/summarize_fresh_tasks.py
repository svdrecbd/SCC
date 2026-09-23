"""Summarize the complete fixed screen; retain dependence-aware uncertainty."""
import json
from pathlib import Path
import sys


def main(directory):
    import numpy as np

    configuration = json.loads((directory / "config.json").read_text())
    records = [json.loads((directory / f"{family}_{seed}" / "result.json").read_text())
               for family in configuration["families"] for seed in configuration["seeds"]]
    names = list(records[0]["scores"])
    aggregate = {name: float(np.mean([record["scores"][name] for record in records])) for name in names}
    public_names = [name for name in names if name not in ("contextual_model", "known_law", "context_frequency", "public_selected")]
    best_public = min(public_names, key=aggregate.get)
    comparisons = {}
    generator = np.random.default_rng(29805)
    bootstrap_indices = generator.integers(len(records), size=(10000, len(records)))
    seed_indices = generator.integers(len(configuration["seeds"]), size=(10000, len(configuration["seeds"])))
    for name in ("public_selected", best_public):
        gains = np.array([record["scores"][name] - record["scores"]["contextual_model"] for record in records])
        seed_means = gains.reshape(len(configuration["families"]), len(configuration["seeds"])).mean(axis=0)
        comparisons[name] = {"absolute_gain": float(gains.mean()), "relative_gain": float(gains.mean()/aggregate[name]),
                             "task_bootstrap_interval": np.quantile(gains[bootstrap_indices].mean(axis=1), [0.025, 0.975]).tolist(),
                             "seed_block_bootstrap_interval": np.quantile(seed_means[seed_indices].mean(axis=1), [0.025, 0.975]).tolist()}
    families = {}
    for family in configuration["families"]:
        subset = [record for record in records if record["family"] == family]
        families[family] = {name: float(np.mean([record["scores"][name] for record in subset])) for name in ("contextual_model", "public_selected", best_public, "known_law")}
    selected_wins = sum(value["contextual_model"] < value["public_selected"] for value in families.values())
    best_wins = sum(value["contextual_model"] < value[best_public] for value in families.values())
    admission = all(value["relative_gain"] >= configuration["required_relative_brier_gain"]
                    and value["task_bootstrap_interval"][0] > 0
                    and value["seed_block_bootstrap_interval"][0] > 0 for value in comparisons.values())
    admission &= min(selected_wins, best_wins) >= configuration["required_family_wins"]
    result = {"tasks": len(records), "queries": len(records)*configuration["query_count"], "aggregate_scores": aggregate,
              "best_aggregate_public_method": best_public, "comparisons": comparisons, "family_scores": families,
              "family_wins_against_selected": selected_wins, "family_wins_against_best_aggregate": best_wins,
              "computational_screen_passed": bool(admission),
              "mean_contextual_model_seconds": float(np.mean([record["contextual_model_seconds"] for record in records])),
              "mean_public_portfolio_seconds": float(np.mean([record["public_portfolio_seconds"] for record in records])),
              "minimum_pointwise_recovery_margin": min(record["minimum_recovery_margin"] for record in records),
              "mean_summed_risk_gain": float(np.mean([record["mean_summed_risk_gain"] for record in records])),
              "neural_training": False}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
