from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent import graph


def test_supervisor_returns_structured_decision(monkeypatch):
    class FakeStructuredLLM:
        def invoke(self, messages):
            assert messages[-1].content == "Calculate POT risk"
            return graph.SupervisorDecision(
                intent="pot",
                alpha=0.99,
                threshold_quantile=0.90,
                plot=True,
            )

    class FakeLLM:
        def with_structured_output(self, model):
            assert model is graph.SupervisorDecision
            return FakeStructuredLLM()

    monkeypatch.setattr(graph, "_get_llm", lambda: FakeLLM())

    result = graph._supervisor(
        {"messages": [{"role": "user", "content": "Calculate POT risk"}]}
    )

    assert result["_intent"] == "pot"
    assert result["_parameters"]["alpha"] == pytest.approx(0.99)
    assert result["_parameters"]["threshold_quantile"] == pytest.approx(0.90)
    assert result["_parameters"]["plot"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("intent", "unsupported"),
        ("alpha", 1.05),
        ("alpha", -0.5),
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("threshold_quantile", 1.0),
        ("block_size", 1),
        ("block_count", 2),
        ("threshold", 0.0),
    ],
)
def test_supervisor_decision_rejects_invalid_parameters(field, value):
    parameters = {
        "intent": "empirical",
        "alpha": 0.99,
        "threshold_quantile": 0.90,
        "block_size": 10,
        "block_count": None,
        "threshold": None,
    }
    parameters[field] = value

    with pytest.raises(ValidationError):
        graph.SupervisorDecision(**parameters)


def test_supervisor_decision_uses_safe_defaults():
    decision = graph.SupervisorDecision(intent="empirical", alpha=0.99)

    assert decision.threshold_quantile == pytest.approx(0.90)
    assert decision.block_size == 10
    assert decision.block_count is None
    assert decision.plot is False
