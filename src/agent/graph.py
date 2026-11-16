"""Shared state, nodes, and LangGraph workflow."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Literal
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import tools
from .prompts import REPORT_AGENT_PROMPT, SUPERVISOR_PROMPT


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: list[dict[str, str]]
    df_path: str
    raw_results: dict[str, Any]
    plot_path: str
    _dataframe: Any                         # resultado de cargar el CSV
    _losses: Any                            # vector de pérdidas
    _intent: str                            # método elegido
    _tool_requests: list[dict[str, Any]]    # herramientas pendientes
    _parameters: dict[str, Any]             # alpha, umbral, block_size, etc.
    _phase: str                             # distingue fase de datos/cálculo
    _plot_done: bool                        # evita generar el gráfico dos veces


def _conversation_messages(state: AgentState) -> list[HumanMessage | AIMessage]:
    """Convert the stored chat history to LangChain messages."""
    messages: list[HumanMessage | AIMessage] = []
    for message in state.get("messages", []):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
        elif message.get("role") in {"user", "human"}:
            messages.append(HumanMessage(content=content))
    return messages


class SupervisorDecision(BaseModel):
    """Structured decision extracted from the user's request by the LLM."""

    intent: Literal["empirical", "pot", "block"] = Field(
        description="Requested risk method: empirical, POT, or block maxima."
    )
    alpha: float = Field(
        gt=0,
        lt=1,
        description=(
            "Confidence level as a decimal. Prefer the user's explicit value; "
            "otherwise use the configured sidebar value."
        ),
    )
    threshold_quantile: float = Field(
        default=0.90,
        gt=0,
        lt=1,
        description=(
            "POT threshold quantile as a decimal. Use 0.90 by default "
            "and always keep it strictly below alpha."
        ),
    )
    block_size: int = Field(
        default=10,
        ge=2,
        description=(
            "Number of observations per block for block maxima. "
            "Use at least 2; default to 10 when the user does not specify it."
        ),
    )
    block_count: int | None = Field(
        default=None,
        ge=3,
        description=(
            "Requested number of blocks. Set this when the user asks for "
            "a specific number of blocks."
        ),
    )
    plot: bool = Field(description="Whether the user requested a diagnostic plot.")


def _get_llm() -> ChatOpenAI:
    """Create the configured OpenAI chat model once per graph process."""
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def _supervisor(state: AgentState) -> dict[str, Any]:
    """Ask the LLM to classify the request and extract its parameters."""
    decision = _get_llm().with_structured_output(SupervisorDecision).invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *_conversation_messages(state),
        ]
    )
    return {
        "_intent": decision.intent,
        "_parameters": decision.model_dump(
            exclude={"intent"},
        ),
    }


def _data_agent(state: AgentState) -> dict[str, Any]:
    """Prepare the uploaded data and schedule the corresponding tool calls."""
    csv_path = state.get("df_path", "")
    if not csv_path:
        raise ValueError("A historical CSV file is required before running an analysis.")
    return {
        "_phase": "data",
        "_tool_requests": [
            {"name": "load_and_clean_csv", "arguments": {"csv_path": csv_path}},
        ]
    }


def _quantitative_agent(state: AgentState) -> dict[str, Any]:
    """Schedule the requested statistical calculation."""
    intent = state.get("_intent", "empirical")
    parameters = state.get("_parameters", {})
    if state.get("_dataframe") is None:
        raise RuntimeError("The data agent did not produce a cleaned DataFrame.")

    if intent == "pot":
        confidence = float(parameters["alpha"])
        threshold_quantile = float(
            parameters.get("threshold_quantile", 0.90)
        )
        if threshold_quantile >= confidence:
            threshold_quantile = confidence * 0.90
        request = {
            "name": "apply_pot_method",
            "arguments": {
                "alpha": confidence,
                "threshold_quantile": threshold_quantile,
            },
        }
    elif intent == "block":
        block_count = parameters.get("block_count")
        if block_count is not None:
            request_arguments = {"block_count": int(block_count)}
        else:
            block_size = max(2, int(parameters.get("block_size", 10)))
            request_arguments = {"block_size": block_size}
        request = {
            "name": "apply_block_maxima",
            "arguments": request_arguments,
        }
    else:
        request = {
            "name": "calculate_empirical_risk",
            "arguments": {"alpha": parameters["alpha"]},
        }
    return {
        "_phase": "quantitative",
        "_tool_requests": [
            {"name": "calculate_loss_vector", "arguments": {}},
            request,
        ],
    }


def _tools_node(state: AgentState) -> dict[str, Any]:
    """Execute pending tool requests and put their outputs back in the state."""
    results = dict(state.get("raw_results", {}))
    dataframe = state.get("_dataframe")
    losses = state.get("_losses")
    plot_path = state.get("plot_path", "")

    for request in state.get("_tool_requests", []):
        name = request["name"]
        arguments = dict(request.get("arguments", {}))
        if name == "load_and_clean_csv":
            dataframe = tools.load_and_clean_csv(**arguments)
        elif name == "calculate_loss_vector":
            if dataframe is None:
                raise RuntimeError("Cannot calculate losses before loading the CSV.")
            losses = tools.calculate_loss_vector(dataframe)
        elif name == "calculate_empirical_risk":
            results.update(tools.calculate_empirical_risk(losses, **arguments))
        elif name == "apply_pot_method":
            results.update(tools.apply_pot_method(losses, **arguments))
        elif name == "apply_block_maxima":
            results.update(tools.apply_block_maxima(losses, **arguments))
        elif name == "plot_tail_diagnostics":
            plot_path = tools.plot_tail_diagnostics(losses, **arguments)
        else:
            raise ValueError(f"Unknown tool requested by the graph: {name}")

    if (
        state.get("_phase") == "quantitative"
        and state.get("_intent") == "pot"
        and state.get("_parameters", {}).get("plot")
        and not state.get("_plot_done", False)
    ):
        threshold = results.get("threshold", results.get("u"))
        xi = results.get("xi")
        beta = results.get("beta")
        if threshold is not None and xi is not None and beta is not None:
            plot_path = tools.plot_tail_diagnostics(
                losses, threshold=threshold, xi=xi, beta=beta
            )

    return {
        "_dataframe": dataframe,
        "_losses": losses,
        "raw_results": results,
        "plot_path": plot_path,
        "_tool_requests": [],
        "_plot_done": bool(plot_path),
    }


def _report_agent(state: AgentState) -> dict[str, Any]:
    """Ask the LLM to write the business-facing report."""
    report_response = _get_llm().invoke(
        [
            SystemMessage(content=REPORT_AGENT_PROMPT),
            *_conversation_messages(state),
            HumanMessage(
                content=json.dumps(
                    {
                        "method": state.get("_intent", "empirical"),
                        "parameters": state.get("_parameters", {}),
                        "results": state.get("raw_results", {}),
                        "plot_path": state.get("plot_path", ""),
                    },
                    default=str,
                )
            ),
        ]
    )
    report = report_response.content
    if not isinstance(report, str):
        raise TypeError("The report agent returned non-text content.")
    messages = [*state.get("messages", []), {"role": "assistant", "content": report}]
    return {"messages": messages[-10:]}


def _after_tools(state: AgentState) -> str:
    """Choose the next stage after the shared executor has run."""
    return "quantitative_agent" if state.get("_phase") == "data" else "report_agent"


def build_graph() -> Any:
    """Build and compile the supervisor/data/quantitative/report workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("data_agent", _data_agent)
    graph.add_node("quantitative_agent", _quantitative_agent)
    graph.add_node("tools", _tools_node)
    graph.add_node("report_agent", _report_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "data_agent")
    graph.add_edge("data_agent", "tools")
    graph.add_conditional_edges(
        "tools",
        _after_tools,
        {"quantitative_agent": "quantitative_agent", "report_agent": "report_agent"},
    )
    graph.add_edge("quantitative_agent", "tools")
    graph.add_edge("report_agent", END)
    return graph.compile()
