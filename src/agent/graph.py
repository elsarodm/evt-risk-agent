"""Shared state, nodes, and LangGraph workflow."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, FiniteFloat

from . import tools
from .prompts import REPORT_AGENT_PROMPT, SUPERVISOR_PROMPT


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: list[dict[str, str]]  # Messages history
    df_path: str                    # Path to the uploaded CSV file
    raw_results: dict[str, Any]     # Quantitative results
    plot_path: str                  # Path to the generated plot
    _intent: str                    # Selected risk method
    _parameters: dict[str, Any]     # Parameters for the selected method
    _losses_cache: Any              # Processed loss vector
    _cached_df_path: str            # Path to the CSV where _losses_cache is stored

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
        description="Requested risk method: 'empirical', 'POT', or 'block'."
    )
    alpha: float = Field(
        gt=0, lt=1,
        description="Confidence level as a decimal (Ej. 0.95 or 0.99)."
    )
    threshold_quantile: float = Field(
        default=0.90, gt=0, lt=1,
        description="POT threshold quantile as a decimal (0.90 by default)."
    )
    threshold: FiniteFloat | None = Field(
        default=None, gt=0,
        description="POT threshold as an absolute decimal loss level if the user specifies a threshold."
    )
    block_size: int = Field(
        default=10, ge=2,
        description="Number of observations per block for block maxima (10 by default)."
    )
    block_count: int | None = Field(
        default=None, ge=3,
        description="Requested number of blocks."
    )
    plot: bool = Field(
        default=False,
        description="Whether the user requested a diagnostic plot."
    )


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
        "_parameters": decision.model_dump(exclude={"intent"}),
    }


def _data_agent(state: AgentState) -> dict[str, Any]:
    """Prepare the uploaded data and calculate the loss vector."""
    cached_csv = state.get("_cached_df_path", "")
    current_csv = state.get("df_path", "")
    if not current_csv:
        raise ValueError("A historical CSV file is required before running an analysis.")

    # If losses are already calculated from the current CSV
    if state.get("_losses_cache") is not None and current_csv == cached_csv:
        return {}

    # If the current CSV is a new file
    df = tools.load_and_clean_csv(current_csv)
    losses = tools.calculate_loss_vector(df)
    
    return {
        "_losses_cache": losses,
        "_cached_df_path": current_csv,
    }


def _quantitative_agent(state: AgentState) -> dict[str, Any]:
    """Execute the requested statistical calculation."""
    intent = state.get("_intent", "empirical")
    params = state.get("_parameters", {})
    losses = state.get("_losses_cache")
    if losses is None:
        raise RuntimeError("The data agent did not produce a loss vector.")

    results: dict[str, Any] = {} 
    plot_path: str = "" 
    
    # Execution of the selected method 
    if intent == "pot": 
        alpha = params["alpha"] 
        threshold = params.get("threshold")
        threshold_quant = params.get("threshold_quantile", 0.90)
        results = tools.apply_pot_method(
            losses,
            alpha=alpha,
            threshold=threshold,
            threshold_quantile=threshold_quant,
        )
        
        # Create plot if requested
        if params.get("plot", False): 
            plot_path = tools.plot_tail_diagnostics(
                losses,
                threshold=results.get("threshold"),
                xi=results.get("xi"),
                beta=results.get("beta"),
            )

    elif intent == "block":
        if params.get("block_count") is not None:
            results = tools.apply_block_maxima(
                losses, 
                block_count=params["block_count"],
                alpha=params["alpha"],
            )
        else:
            results = tools.apply_block_maxima(
                losses, 
                block_size=params.get("block_size", 10),
                alpha=params["alpha"],
            )
        if params.get("plot", False):
            plot_path = tools.plot_block_diagnostics(
                losses,
                block_maxima=results["block_maxima"],
                xi=results["xi"],
                mu=results["mu"],
                sigma=results["sigma"],
            )
    
    else:  # empirical
        results = tools.calculate_empirical_risk(
            losses, 
            alpha=params["alpha"]
        )
    return {
        "raw_results": results,
        "plot_path": plot_path,
    }


def _report_agent(state: AgentState) -> dict[str, Any]:
    """Ask the LLM to write the business-facing report."""
    latest_user_request = next(
        (
            message["content"]
            for message in reversed(state.get("messages", []))
            if message.get("role") in {"user", "human"}
            and isinstance(message.get("content"), str)
        ),
        "",
    )
    report_context = json.dumps(
        {
            "user_request": latest_user_request,
            "method": state.get("_intent", "empirical"),
            "parameters": state.get("_parameters", {}),
            "results": state.get("raw_results", {}),
            "has_plot": bool(state.get("plot_path")),
        },
        default=str,
    )

    response = _get_llm().invoke(
        [
            SystemMessage(content=REPORT_AGENT_PROMPT),
            HumanMessage(content=report_context),
        ]
    )
    report_content = response.content
    if not isinstance(report_content, str):
        raise TypeError("The report agent returned non-text content.")
    messages = [*state.get("messages", []), {"role": "assistant", "content": report_content}]
    return {"messages": messages[-10:]}


def build_graph() -> Any:
    """Build and compile the supervisor/data/quantitative/report workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", _supervisor)
    graph.add_node("data_agent", _data_agent)
    graph.add_node("quantitative_agent", _quantitative_agent)
    graph.add_node("report_agent", _report_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "data_agent")
    graph.add_edge("data_agent", "quantitative_agent")
    graph.add_edge("quantitative_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()
