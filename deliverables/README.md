# Dated deliverables

These are preserved sharing artifacts, not parallel current research plans.
Start with the [current labnotes](../labnotes.md#current-position) for subsequent
results and corrections.

| Prepared | Package | Scope |
|---|---|---|
| 20 September | [Safety–Capability Coupling Whitepaper](../output/pdf/Safety_Capability_Coupling_Whitepaper.pdf) · [editable source](scc-whitepaper/Safety_Capability_Coupling_Whitepaper.md) | Mathematical synthesis through LN-239, claim boundaries, proof appendix and evidence map |
| 16 September | [Evidence figures](scc-figures-20260916-v1/README.md) | Five figures, PDF, editable SVGs, data and reproducible plot source |
| 15 September | [Theory v4.1](scc-theory-frontier-20260915/SCC_Theory_and_Editable_Model_Bridge_v4.md) | Conditional theorem and editable-model boundary; retains its v4 filename |
| 14 September | [Theorem research proposal](scc-theory-pitch-20260914/SCC_Theorem_Research_Proposal.md) | Historical proposal, not authorization for its experiments |
| 10 September | [Original master document](scc-master-20260910-v1/SCC_Master_Document.md) | Early synthesis; later labnotes supersede its assessment |

The [derived v4 bridge](scc-theory-frontier-20260915/SCC_Editable_Model_Bridge_v4.md)
and [validation receipt](scc-theory-frontier-20260915/v4_validation_receipt.txt)
belong to that dated review. None of these packages demonstrates the full SCC
mechanism. Original experimental inputs and checkpoints live in the evidence store.

The whitepaper is rendered from its Markdown source by
[`scripts/render_whitepaper.py`](../scripts/render_whitepaper.py), using ReportLab,
Pandoc, XeLaTeX with TeX Gyre fonts, and pypdf. Supply `--work-directory` with an
intermediate-output directory; the final PDF is written under `output/pdf/`.
