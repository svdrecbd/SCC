#!/usr/bin/env python3
"""Static integration audit for canonical SCC theory Version 4.

This verifies synchronization and internal document contracts. It is not a proof
assistant and does not replace the symbolic arguments or finite exact checkers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables" / "scc-theory-frontier-20260915"
CANONICAL = DELIVERABLES / "SCC_Theory_and_Editable_Model_Bridge_v4.md"
BRIDGE = DELIVERABLES / "SCC_Editable_Model_Bridge_v4.md"
DERIVER = ROOT / "scripts" / "derive_scc_bridge_note.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_deriver():
    spec = importlib.util.spec_from_file_location("derive_scc_bridge_note", DERIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load bridge derivation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_characters(text: str) -> list[dict[str, int]]:
    return [
        {"index": index, "codepoint": ord(char)}
        for index, char in enumerate(text)
        if ord(char) < 32 and char not in "\n\r\t"
    ]


def audit() -> dict[str, object]:
    text = CANONICAL.read_text(encoding="utf-8")
    bridge_text = BRIDGE.read_text(encoding="utf-8")
    deriver = load_deriver()

    section_pairs = re.findall(r"^## (\d+)\.\s+(.+)$", text, re.MULTILINE)
    section_numbers = [int(number) for number, _ in section_pairs]
    expected_sections = list(range(1, 22))

    canonical_refs = re.findall(r"Section(?:s)?\s+([0-9]+(?:\.[0-9]+)?)", text)
    canonical_missing = sorted(
        {
            ref
            for ref in canonical_refs
            if int(ref.split(".")[0]) not in set(section_numbers)
        }
    )

    bridge_body = text.split(deriver.START, 1)[1].split(deriver.END, 1)[0]
    bridge_sections = {
        int(number)
        for number in re.findall(r"^## (\d+)\.\s+", bridge_body, re.MULTILINE)
    }
    bridge_refs = re.findall(
        r"Section(?:s)?\s+([0-9]+(?:\.[0-9]+)?)", bridge_body
    )
    bridge_missing = sorted(
        {
            ref
            for ref in bridge_refs
            if int(ref.split(".")[0]) not in bridge_sections
        }
    )

    theorem_labels = set(
        re.findall(
            r"^### (?:Lemma|Theorem|Proposition|Corollary)\s+([0-9]+(?:\.[0-9]+)?)",
            text,
            re.MULTILINE,
        )
    )

    required_fragments = {
        "attack_specific_prefix": (
            "Fix an allowed attack \\(A\\), chosen before the task instance, "
            "and let \\(p_A=\\Pr_{P_A}(V_A)\\). If \\(p_A\\ge p_0>0\\)"
        ),
        "sharp_conditioning": "\\frac{\\varepsilon}{\\max\\{p,q\\}}",
        "null_ideal_branch": "If \\(q_A=0\\), then \\(p_A\\le\\varepsilon_A\\)",
        "separate_nonvacuity": "## 7. Separate nonvacuity condition",
        "fixed_attack_resource_quantifier": "The choice of \\(A\\) is outside the probability",
        "instancewise_distinct": "### 8.2 Stronger instancewise object",
        "preservation_baseline": "\\min\\{p_A,p_A\\beta_b+\\varepsilon_A\\}",
        "disclosure_joint_bound": "p_A G_b^{\\mathrm{disc}}(\\mu)+\\varepsilon_A",
        "selected_disclosure_baseline": "p_A\\Gamma_{A,V}^{\\mathrm{disc}}+\\varepsilon_A",
        "original_action_unconditional_bound": "G_0^{\\mathrm{disc}}(\\mu)+\\varepsilon_A",
        "recovery_query_information": "before any task-dependent query side information is provided",
        "reset_evaluation_scope": "both table construction and scored evaluation reset",
        "exact_leakage_alphabet": "|\\mathcal L_A|\\le2^\\ell",
        "relational_complement": "### Proposition 8.1 — Encoder-free dense complement rewrite",
        "two_policy_bit_escape": "### Proposition 12 — Two-policy-bit task-preserving escape",
        "broad_security_assumed": "For a broad resource family this is an **assumed security property**",
        "intrinsic_open": "Intrinsic learned SCC remains the central open problem.",
    }
    required_present = {
        name: fragment in text for name, fragment in required_fragments.items()
    }

    forbidden_fragments = [
        "Nine regression tests covering both scripts pass",
        "Nine new regression tests pass",
        "Possible with selection and baseline degradation",
        "or a commit after every modification",
        "\\frac{2\\varepsilon_A}{p_A}",
    ]
    forbidden_present = {
        fragment: fragment in text for fragment in forbidden_fragments
    }

    expected_bridge = deriver.render(CANONICAL)
    bridge_synchronized = bridge_text == expected_bridge
    bad_controls = control_characters(text)

    passed = all(
        [
            section_numbers == expected_sections,
            not canonical_missing,
            not bridge_missing,
            text.count(deriver.START) == 1,
            text.count(deriver.END) == 1,
            all(required_present.values()),
            not any(forbidden_present.values()),
            bridge_synchronized,
            not bad_controls,
        ]
    )

    result: dict[str, object] = {
        "canonical": str(CANONICAL.relative_to(ROOT)),
        "canonical_sha256": sha256(CANONICAL),
        "derived_bridge": str(BRIDGE.relative_to(ROOT)),
        "derived_bridge_sha256": sha256(BRIDGE),
        "numbered_sections": section_numbers,
        "expected_numbered_sections": expected_sections,
        "canonical_section_references_checked": len(canonical_refs),
        "missing_canonical_section_references": canonical_missing,
        "bridge_section_references_checked": len(bridge_refs),
        "missing_bridge_section_references": bridge_missing,
        "theorem_labels": sorted(theorem_labels),
        "bridge_start_markers": text.count(deriver.START),
        "bridge_end_markers": text.count(deriver.END),
        "bridge_synchronized": bridge_synchronized,
        "required_fragments_present": required_present,
        "forbidden_fragments_present": forbidden_present,
        "control_characters": bad_controls,
        "scope_note": (
            "Static integration only; symbolic proofs and exact finite examples "
            "are validated separately."
        ),
        "passed": passed,
    }
    if not passed:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
