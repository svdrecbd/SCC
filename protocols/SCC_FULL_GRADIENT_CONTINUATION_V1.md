# SCC full-gradient continuation

Development continuation, 2026-09-10. The short pilot's control and both
candidates passed the intact gate. Stock-Adam removal/recovery still retained
substantial cognition. The stronger candidate had lower average training
penalties on the same final20 meta episodes (.61474 versus .70984 for weak
coupling), without a material clean task loss (minimum99.22%). This motivates
one longer continuation; it is not positive SCC evidence or proof of convergence.

Continue the preserved pilot control and weight1 candidate for4,000 additional
ordinary updates each. Preserve their respective optimizer moments, RNG and
ordinary-stream state. Match ordinary batches and retain all previous source
and checkpoints. This is a new continuation contract, not a mutation of the
pilot's1,000-step contract. Coupling settings remain unchanged: weight1,
full stabilized-Adam derivative, epsilon1e-4, mean13-domain deficit, one episode
every10 ordinary steps. Use ordinals20100..20499, giving500 total meta episodes
including the initial pilot. The weak arm is not extended.

At the parent and every1,000 updates, report the same held development meta
episodes1100..1103 and32-problem intact diagnostics. These fixed episodes do
not train the model. They remain train-split samples and may overlap lifetime
ordinary training; they measure optimizer-objective generalization across new
episodes, not sealed scientific generalization. Final qualification still uses
the original128-problem suite in both layouts and all fixed text blocks.

At completion, repeat original stock-Adam epsilon1e-8 removal/recovery and the
preselected component lesions/counterfactual activation replacements. Save every
stage even if a parent fails qualification, labeled diagnostic-only. Preserve
all intermediate and final checkpoints. Do not describe decreases in the
confidence-based objective, ordinary component damage, or a failed bounded
search as a successful SCC mechanism or an impossibility result.

Check optimizer-preserving continuation against a continuous CPU fixture before
submission. The GPU job ceiling is20H100-minutes (about $1), allowing headroom
for both arms, checkpoint diagnostics and final probes, within the existing
$10 aggregate development limit. The previous work has cost $2.59077.
