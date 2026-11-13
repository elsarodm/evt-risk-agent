"""Shared state, nodes, and LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: list[dict[str, str]]
    df_path: str
    raw_results: dict[str, Any]
    plot_path: str


def build_graph() -> Any:
    """Build the agent graph.

    Node and edge definitions will be added here following the
    supervisor--data--quantitative--tools--report workflow.
    """
    raise NotImplementedError
