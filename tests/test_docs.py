"""Docs are generated: every number comes from results/*.json via docs/build_docs.py."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

TEMPLATES = ["docs/proposal.template.md", "docs/README.template.md"]


def _prose(text: str) -> str:
    text = re.split(r"(?m)^## (?:\d+\. )?References", text)[0]      # citations carry years and volumes
    text = re.sub(r"```.*?```", "", text, flags=re.S)              # code blocks
    text = re.sub(r"\{\{.*?\}\}", "", text)                         # placeholders
    text = re.sub(r"\]\([^)]*\)", "]", text)                        # link targets
    text = re.sub(r"(?m)^#+ [\d.]+", "", text)                      # numbered headings
    text = re.sub(r"Section \d+(\.\d+)?", "", text)                  # cross-references
    return text


@pytest.mark.parametrize("tpl", TEMPLATES)
def test_templates_carry_no_hand_typed_results(tpl):
    prose = _prose((ROOT / tpl).read_text(encoding="utf-8"))
    bad = [m.group(0) for m in re.finditer(r"\d+\.\d+|\b\d{3,}\b", prose)
           if not re.fullmatch(r"(19|20)\d\d", m.group(0))]
    assert not bad, f"numbers typed into {tpl}: {bad}"


def test_rendered_docs_are_up_to_date():
    pytest.importorskip("matplotlib")
    import build_docs
    for dst, text in build_docs.build(write=False).items():
        assert (ROOT / dst).read_text(encoding="utf-8") == text, f"{dst} is stale: run python docs/build_docs.py"

