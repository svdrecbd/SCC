# SCC compute budget assessment — 9 September 2026

This is a planning assessment, not a change to the research protocol, a runtime
benchmark, or authorization to spend credits. No GPU was started for this assessment.
Every experimental model can originate from random initialization under our own
training pipeline. Existing open-source training software and public data are
compatible with that requirement; their versions and provenance must be recorded.

## Verified access and rates

- The installed GiveMeANode CLI authenticates successfully.
- Its billing view reports active status, a **$53.88 monthly cap on the default
  workspace**, and $0.00 month-to-date spend for this CLI identity. This is a cap,
  not the credit balance or a verified remaining workspace allowance.
- The identity has member permissions. Organization credit balances are visible
  only to admin/billing roles. The browser billing page also requires sign-in.
  **$450 remains the user's reported available credit**, not an independently
  verified balance. Other workspace members' usage can reduce available headroom.
- A read-only node inspection quotes $3.996 per H100 GPU-hour for interactive use.
- The public price table quotes **$2.997 per H100 GPU-hour for batch jobs**.
- Friday's rate is $2.67732 per GPU-hour. Discounts do not stack: this is about
  10.7% cheaper than the normal batch price, not a further 33% off batch.
- $450 buys about **150.15 batch GPU-hours**, or **18.77 hours on eight GPUs**,
  before other charges. The $53.88 cap is equivalent to at most about 17.98 batch
  GPU-hours if the workspace has no other spend or limiting cap.

Sources: [GiveMeANode pricing](https://givemeanode.com/),
[documentation](https://givemeanode.com/docs), and read-only local CLI calls
`gman whoami`, `gman billing`, `gman node ls`, `gman node get`, and `gman limits`.

## Calculation and scope

The calculator is `budget/estimate.py`. It only performs local arithmetic.

For parameters N and training tokens D:

    baseline FLOPs ≈ 6 N D
    baseline GPU-hours = baseline FLOPs / (effective TFLOP/s × 10^12 × 3600)
    baseline cost = GPU-hours × $2.997

Effective throughput includes normal training overhead relative to the 6ND proxy.
It is NOT advertised peak throughput. Assumed ranges are:

| Parameters | Tokens per run | Assumed effective TFLOP/s per H100 | Ordinary run cost |
|---|---:|---:|---:|
| 125M, optional smaller study | 5B | 75–200 | $16–42 |
| 350M | 10B | 150–300 | $58–117 |
| 1B | 20B | 200–400 | $250–500 |
| 3B | 50B | 250–450 | $1,665–2,997 |
| 7B | 100B | 250–450 | $7,770–13,986 |

These token budgets use the examples in `04_SCALING_LADDER.md`. They are research
budgets, not guarantees of modern assistant-level ability. Cost scales approximately
linearly with tokens at fixed architecture, context, and throughput. A 7B run at
1T tokens costs approximately ten times its 100B-token counterpart.

Published efficient implementations provide a plausibility check, not an SCC
benchmark: [llm.c's 1.6B reproduction](https://github.com/karpathy/llm.c/discussions/677)
reports roughly 24 hours on eight H100s for its specific run. NVIDIA's
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) documents high utilization with
optimized training. Our modified trainer may be substantially slower, especially
with differentiable inner loops. Do not transfer those benchmarks as promises.

The replicated-study scenario uses **three arms × three independent seeds**:
ordinary LM, invariant-only LM, and SCC LM. For each seed, SCC costs an assumed
2–5 ordinary training runs. We then add 50% for limited attack, evaluation,
recovery, and tuning work, and 25% contingency. This yields 22.5–39.375 ordinary-run
equivalents per scale, not merely nine ordinary runs.

This is a bounded research campaign. It does not price an unlimited mechanism
search, every architecture variant, a full additional early-versus-late arm at
every scale, or enough attacks to prove retraining-level escape costs. Extra
comparison arms need explicit allocation once chosen. Training overhead above
5× or extensive recovery attacks can push cost past the high estimate. The ranges
are paired planning scenarios, not statistical confidence intervals or ceilings.

## Cumulative budget for the documented ladder

Includes a $150–450 toy-pilot allocation, followed by the 350M, 1B, 3B, and 7B
studies in order. The optional 125M study is not added to this ladder.

| Complete through | Modeled cumulative compute budget | Additional funds beyond $450 |
|---|---:|---:|
| Toy pilot, 20–80M | $150–450 | $0 if credits and cap permit |
| 350M mechanism study | $1,461–5,039 | $1,011–4,589 |
| 1B robustness study | $7,081–24,707 | $6,631–24,257 |
| 3B scaling study | $44,543–142,714 | $44,093–142,264 |
| 7B capstone | $219,368–693,413 | $218,918–692,963 |

For a practical initial funding target, round up to **$2,000–6,000 total** for
the pilot and 350M stage, allowing additional small-scale comparator work. That
means **$1,550–5,550 more than the reported $450**. Approximately $5,000 total
($4,550 additional) is a reasonable provisional target, to revise after profiling.

A leaner full ladder retaining three seeds at 350M/1B, two at 3B, and only one at
7B costs approximately **$90,331–286,944**. The capstone would then be a single-seed
validation, not evidence of seed robustness at that scale. This is a different
evidence standard, not a computational shortcut with unchanged scientific strength.

The optional 125M/5B-token replicated study alone is modeled at $351–1,639,
before the toy-pilot allocation. $450 could approach its favorable case, but is
not a dependable budget for completing it plus debugging and broader comparisons.

These are compute-centered estimates. They exclude researcher salaries, paid
datasets, external reviewers, and paid model/API labels. Storage and preprocessing
must be budgeted within the reserve at small scale and separately once measured at
large scale. No successful scientific outcome is guaranteed by a spending level.

## A bounded use of the initial $450

Keep data preparation, code development, manifests, and synthetic generators local
where practical. A provisional allocation is:

| Work | Spending allocation |
|---|---:|
| Benchmark baseline/SCC throughput, memory, and checkpoint resume | $30 |
| 20–80M ordinary/invariant/SCC pilot, preferably paired across seeds | $180 |
| Capability-preserving attacks, recovery attempts, benign edits | $150 |
| Reruns, preprocessing, storage, and unexpected overhead | $90 |
| Total | $450 |

The allocations are spending limits for a future approved plan, not claims that
a specified number of runs already fits. After profiling, set token budgets and
run counts to fit them. The result should be a working audited pipeline and an
initial coupling assessment; it need not establish the Tier-1 claim.

## Platform details that affect the plan

The documented hardware is H100 SXM with 80 GB per GPU, with NVLink inside the
eight-GPU shape. A single eight-GPU machine is adequate hardware to investigate
sub-10B dense models using suitable sharding; the SCC inner-loop memory footprint
still needs measurement. More GPUs reduce elapsed time but do not eliminate total
GPU-hours, and small models may run less efficiently when spread across eight GPUs.

The live default limits include a 12-hour maximum batch-attempt duration, 100 GiB
job scratch, and a 50 GiB checkpoint slot. A conventional mixed-precision 7B Adam
training state can exceed that checkpoint allowance. Large runs therefore require
an explicit durable resume/export design and potentially higher storage limits.
Checkpoints must preserve optimizer, scheduler, RNG, and data-position state.

The visible $53.88 monthly workspace cap must be reconciled with the intended
project allowance before attempting a $450 campaign. Raising a cap does not add
credits. No caps, credentials, nodes, or jobs were changed in this assessment.

## Evidence to collect before trusting the estimate

1. Sustained tokens/second for the exact baseline and SCC objective after warmup.
2. Peak GPU memory, including inner-loop copies and optimizer state.
3. Actual SCC/baseline runtime ratio at matched outer training tokens.
4. End-to-end billed GPU-hours including evaluation and checkpoint overhead.
5. A restart that reproduces the intended data and optimizer trajectory.
6. Versioned code, tokenizer, data manifests, seeds, raw metrics, and attack logs.

Owning the training lineage supports causal interpretation. Strong review also
depends on the controls, adaptive attacks, held-out evaluation, and reproducibility;
model ownership alone cannot guarantee an unimpeachable result.
