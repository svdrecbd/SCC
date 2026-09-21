# Experiment navigation

These are preserved finite prototypes, repair controls and historical probes.
Read the linked labnotes phase before choosing a runner: a working verifier is
not evidence that its candidate meets the SCC target. CPU jobs run on Charon.

| Area | Implementations | Record |
|---|---|---|
| Checked finite machines and task reductions | [machine](bend_machine/), [reachability](bend_reachability/), [parity screen](bend_parity_screen/) | [LN-153–177](../labnotes.md#phase-153) |
| Fresh learning, priors and missing information | [online learning](bend_online_learning/), [task prior](bend_task_prior/), [conditional relearning](conditional_relearning/), [semantic privacy](semantic_privacy/) | [LN-153–177](../labnotes.md#phase-153) |
| Approximate judgment and constructive recovery | [judgment family](judgment_family/), [parity adapter](parity_recovery_adapter/), [inference replacement](inference_replacement/), [verified recovery](verified_recovery/), [list recovery](list_recovery/) | [LN-178–202](../labnotes.md#phase-178) |
| Planning and contingency recovery | [goal recovery](goal_recovery/), [compiled recovery](compiled_recovery/), [contingency recovery](contingency_recovery/) | [LN-178–202](../labnotes.md#phase-178) |
| Causal and dynamical reconstruction | [predictor recovery](predictor_recovery/), [causal quotient](causal_quotient/), [procedure realization](procedure_realization/), [nonlinear recovery](nonlinear_recovery/) | [LN-203–225](../labnotes.md#phase-203) |
| Observation and acquisition cost | [partial observation](partial_observation/), [copy-aware recovery](copy_aware_recovery/) | [LN-203–225](../labnotes.md#phase-203) |

The loose `*topology*`, gradient and optimizer scripts in this directory reproduce
earlier exploratory probes. They remain with their original code and tests;
[early labnotes](../labnotes.md#phase-001) explain their limits and corrections.

Directories with `qualify.py`/`audit.py` use frozen source, explicit inputs and
fresh output paths. Follow the recorded plan and [operations guide](../docs/OPERATIONS.md)
for toolchains and resource limits. A folder name does not authorize another run.
