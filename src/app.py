"""Streamlit interface for the EVT Risk Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from agent.graph import AgentState, build_graph


DATA_DIRECTORY = Path(__file__).parent / "data"
MAX_MESSAGES = 10


def _initialise_session() -> None:
    """Create the conversation state once for the current browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df_path" not in st.session_state:
        st.session_state.df_path = ""
    if "raw_results" not in st.session_state:
        st.session_state.raw_results = {}
    if "plot_path" not in st.session_state:
        st.session_state.plot_path = ""


def _save_uploaded_file(uploaded_file: Any) -> Path:
    """Save an uploaded CSV outside Streamlit's transient upload buffer."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}_{Path(uploaded_file.name).name}"
    destination = DATA_DIRECTORY / filename
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _normalise_message(message: Any) -> dict[str, str] | None:
    """Convert graph messages and dictionaries to the UI message shape."""
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    else:
        role = getattr(message, "type", None)
        content = getattr(message, "content", None)

    if role == "human":
        role = "user"
    elif role in {"ai", "assistant"}:
        role = "assistant"
    if role not in {"user", "assistant"} or not isinstance(content, str):
        return None
    return {"role": role, "content": content}


def _state_from_session(prompt: str, confidence: float) -> AgentState:
    """Build the state expected by the LangGraph entry point."""
    messages = [*st.session_state.messages[-MAX_MESSAGES:]]
    messages[-1] = {
        "role": "user",
        "content": f"{prompt}\n\n[Configured confidence level: {confidence:.1%}]",
    }
    return {
        "messages": messages,
        "df_path": st.session_state.df_path,
        "raw_results": st.session_state.raw_results,
        "plot_path": st.session_state.plot_path,
    }


def _run_agent(prompt: str, confidence: float) -> None:
    """Run the configured graph and store its returned state in the session."""
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    if not st.session_state.df_path:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please upload a historical CSV file from the sidebar first.",
            }
        )
        return

    try:
        result = build_graph().invoke(_state_from_session(prompt, confidence))
    except NotImplementedError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "The analysis workflow is not available yet. "
                    "The interface received your data successfully."
                ),
            }
        )
        return
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"The analysis could not be run: {error}",
            }
        )
        return

    returned_messages = [
        normalised
        for message in result.get("messages", [])
        if (normalised := _normalise_message(message)) is not None
    ]
    st.session_state.messages = (returned_messages or [
        *st.session_state.messages,
        {
            "role": "assistant",
            "content": "The agent did not return a report message.",
        },
    ])[-MAX_MESSAGES:]
    st.session_state.raw_results = result.get(
        "raw_results", st.session_state.raw_results
    )
    st.session_state.plot_path = result.get("plot_path", st.session_state.plot_path)


def _render_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="EVT Risk Agent", page_icon="📊")
    _initialise_session()

    st.title("EVT Risk Agent")
    st.caption("Market risk analysis using Extreme Value Theory")

    with st.sidebar:
        st.header("Data input")
        uploaded_file = st.file_uploader(
            "Drag and drop a historical CSV with dates and prices",
            type=["csv"],
            help="The file is stored temporarily for analysis.",
        )
        confidence = st.slider(
            "Confidence level",
            min_value=0.9,
            max_value=0.999,
            value=0.99,
            step=0.001,
            format="%g%%",
        )

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.get("uploaded_name"):
                path = _save_uploaded_file(uploaded_file)
                st.session_state.df_path = str(path)
                st.session_state.uploaded_name = uploaded_file.name
                st.success(f"File ready: {uploaded_file.name}")
            else:
                st.success(f"File ready: {uploaded_file.name}")

        if st.session_state.df_path:
            st.caption(f"Ruta temporal: {Path(st.session_state.df_path).name}")

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input(
        "Ask for an analysis, for example: calculate empirical VaR and CVaR at 99%"
    )
    if prompt:
        _run_agent(prompt, confidence)
        st.rerun()

    if st.session_state.plot_path:
        plot_path = Path(st.session_state.plot_path)
        if plot_path.is_file():
            st.image(str(plot_path), caption="Tail diagnostic")
        else:
            st.warning("The agent reported a plot, but the file was not found.")


if __name__ == "__main__":
    main()

