#!/usr/bin/env python3
"""Derive the short SCC editable-model bridge note from canonical Version 4."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

START = "<!-- BRIDGE-NOTE-START -->"
END = "<!-- BRIDGE-NOTE-END -->"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "deliverables"
    / "scc-theory-frontier-20260915"
    / "SCC_Theory_and_Editable_Model_Bridge_v4.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "deliverables"
    / "scc-theory-frontier-20260915"
    / "SCC_Editable_Model_Bridge_v4.md"
)


def render(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("canonical document must contain exactly one bridge marker pair")
    body = text.split(START, 1)[1].split(END, 1)[0].strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (
        "# Safety–Capability Coupling Under Editable Models\n\n"
        "15 September 2026 · Derived Version 4.1 bridge note\n\n"
        "This note is generated from the marked Part I of "
        "`SCC_Theory_and_Editable_Model_Bridge_v4.md`. Edit the canonical file, "
        "not this derivative.\n\n"
        f"Canonical SHA-256 at derivation: `{digest}`\n\n"
        "---\n\n"
        f"{body}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = render(args.source)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit("derived bridge note is out of sync with canonical v4")
        print("derived bridge note is synchronized")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
