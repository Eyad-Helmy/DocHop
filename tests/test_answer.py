import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from codearchguard.cli import main
from codearchguard.rag.answer import ABSTENTION, _check_evidence_ids, generate_answer


def test_ask_reports_missing_key_before_ingestion(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAG_LLM_MODEL", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["ask", str(tmp_path), "shop.Item", "Where is the price?"])
    assert exc.value.code == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_no_evidence_skips_generation():
    assert "No repository evidence" in generate_answer(SimpleNamespace(results=[]), "unused")


@pytest.mark.parametrize("model", ["gemini/gemini-flash-latest", "openrouter/google/example"])
def test_generation_sends_bounded_evidence_and_prints_source(monkeypatch, model):
    message = SimpleNamespace(content="It applies the discount [E1].")
    completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=RuntimeError))
    meta = dict(owner_symbol_id="shop.Item", source_file="shop.py", start_line=4, end_line=7)
    hit = SimpleNamespace(node=SimpleNamespace(metadata=meta, text="x" * 3000))
    result = SimpleNamespace(results=[hit] * 6, query="How?", target_symbol="shop.Item")
    answer = generate_answer(result, model)
    call = completion.call_args.kwargs
    assert call["model"] == model
    excerpts = json.loads(call["messages"][1]["content"])["excerpts"]
    assert len(excerpts) == 5
    assert len(excerpts[0]["text"]) == 2500
    assert "[E1] shop.py:4-7" in answer


@pytest.mark.parametrize("answer", ["An unsupported claim.", "Discount [E1] and tax [E9].", "Discount [E0]."])
def test_invalid_evidence_ids_withhold_answer(answer):
    with pytest.raises(ValueError, match="Answer withheld"):
        _check_evidence_ids(answer, {"E1"})


@pytest.mark.parametrize("answer", [ABSTENTION, "The code applies a multiplier [E1] [E2]."])
def test_accepts_abstention_or_supplied_evidence(answer):
    _check_evidence_ids(answer, {"E1", "E2"})
