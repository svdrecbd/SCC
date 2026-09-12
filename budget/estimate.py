"""SCC planning estimates, not measured runtimes. No cloud operations.

Run: python3 budget/estimate.py
All experimental models are assumed to originate from random initialization.
Edit the explicit assumptions below after measuring the actual trainer.
"""

import json

RATE_USD_PER_GPU_HOUR = 2.997
AVAILABLE_CREDIT_USD = 450.0  # User estimate; not visible to the CLI member role.
PILOT_USD = (150.0, 450.0)
SCC_TRAINING_MULTIPLIER = (2.0, 5.0)
ATTACK_EVAL_AND_SEARCH_FRACTION = 0.50
CONTINGENCY_FRACTION = 0.25

# name, parameters, tokens, slow and fast effective baseline TFLOP/s per GPU
# Effective throughput is an assumption, NOT H100 advertised peak performance.
SCALES = [
    ("125M alternative mechanism study", 125e6, 5e9, 75, 200),
    ("350M", 350e6, 10e9, 150, 300),
    ("1B", 1e9, 20e9, 200, 400),
    ("3B", 3e9, 50e9, 250, 450),
    ("7B", 7e9, 100e9, 250, 450),
]


def estimate():
    cumulative = list(PILOT_USD)
    lean_cumulative = list(PILOT_USD)
    result = []
    for name, parameters, tokens, slow_tflops, fast_tflops in SCALES:
        flops = 6 * parameters * tokens
        hours = [flops / (speed * 1e12 * 3600)
                 for speed in (fast_tflops, slow_tflops)]
        ordinary_cost = [h * RATE_USD_PER_GPU_HOUR for h in hours]
        # Each seed has base LM, invariant-only LM, and SCC LM.
        # The first two cost one ordinary run each; SCC costs m ordinary runs.
        equivalents = [3 * (2 + m) * (1 + ATTACK_EVAL_AND_SEARCH_FRACTION)
                       * (1 + CONTINGENCY_FRACTION)
                       for m in SCC_TRAINING_MULTIPLIER]
        study_cost = [c * e for c, e in zip(ordinary_cost, equivalents)]
        row = {
            "scale": name,
            "parameters": int(parameters),
            "tokens": int(tokens),
            "baseline_flops": flops,
            "baseline_effective_tflops_slow_fast": [slow_tflops, fast_tflops],
            "baseline_gpu_hours_low_high": hours,
            "one_ordinary_run_usd_low_high": ordinary_cost,
            "three_arms_three_seeds_study_usd_low_high": study_cost,
        }
        if name != "125M alternative mechanism study":
            cumulative = [a + b for a, b in zip(cumulative, study_cost)]
            lean_seeds = 1 if name == "7B" else 2 if name == "3B" else 3
            lean_cumulative = [a + b * lean_seeds / 3
                               for a, b in zip(lean_cumulative, study_cost)]
            row.update({
                "cumulative_replicated_usd_low_high": cumulative.copy(),
                "gap_from_450_usd_low_high": [max(0, c - AVAILABLE_CREDIT_USD)
                                             for c in cumulative],
                "lean_seeds_at_this_scale": lean_seeds,
                "cumulative_lean_usd_low_high": lean_cumulative.copy(),
            })
        result.append(row)
    return {
        "as_of": "2026-09-09",
        "rate_usd_per_gpu_hour": RATE_USD_PER_GPU_HOUR,
        "credit_is_user_reported": True,
        "batch_gpu_hours_for_450": AVAILABLE_CREDIT_USD / RATE_USD_PER_GPU_HOUR,
        "assumptions_are_not_confidence_intervals_or_spend_authorization": True,
        "scales": result,
    }


if __name__ == "__main__":
    print(json.dumps(estimate(), indent=2))
