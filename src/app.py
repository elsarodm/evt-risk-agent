"""Streamlit interface for the EVT Risk Agent."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
from openai import APIError

from agent.graph import AgentState, build_graph


DATA_DIRECTORY = Path(__file__).parent / "data"
MAX_MESSAGES = 10


def _initialise_session() -> None:
    """Create the conversation state once for the current browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df_path" not in st.session_state:
        st.session_state.df_path = ""
    if "_losses_cache" not in st.session_state:
        st.session_state._losses_cache = None
    if "_cached_df_path" not in st.session_state:
        st.session_state._cached_df_path = ""
    if "uploaded_hash" not in st.session_state:
        st.session_state.uploaded_hash = ""
    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = ""
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
    normalised = {"role": role, "content": content}
    plot_path = message.get("plot_path") if isinstance(message, dict) else None
    if isinstance(plot_path, str) and plot_path:
        normalised["plot_path"] = plot_path
    return normalised


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
        "_losses_cache": st.session_state._losses_cache,
        "_cached_df_path": st.session_state._cached_df_path,
    }


def _run_agent(prompt: str, confidence: float) -> None:
    """Run the configured graph and store its returned state in the session."""
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.df_path:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please upload a historical CSV file from the sidebar first.",
            }
        )
        return

    try:
        with st.chat_message("assistant"):
            with st.status("Thinking..."):
                result = build_graph().invoke(                  # Call agents
                    _state_from_session(prompt, confidence)
                )
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
    except APIError as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"The OpenAI request failed: {error}",
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
    st.session_state.plot_path = result.get("plot_path", "")
    if st.session_state.plot_path:
        for message in reversed(returned_messages):
            if message["role"] == "assistant":
                message["plot_path"] = st.session_state.plot_path
                break
    st.session_state._losses_cache = result.get(
        "_losses_cache", st.session_state._losses_cache
    )
    st.session_state._cached_df_path = result.get(
        "_cached_df_path", st.session_state._cached_df_path
    )


def _render_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        plot_path_value = message.get("plot_path")
        if plot_path_value:
            plot_path = Path(plot_path_value)
            if plot_path.is_file():
                caption = (
                    "Block-maxima diagnostic"
                    if plot_path.name.startswith("plot_block_")
                    else "POT tail diagnostic"
                )
                st.image(str(plot_path), caption=caption)
            else:
                st.warning(f"The saved plot was not found: {plot_path.name}")


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
            min_value=90.0,
            max_value=99.9,
            value=99.0,
            step=0.1,
            format="%.1f%%",
        )
        confidence /= 100

        if uploaded_file is not None:
            uploaded_hash = sha256(uploaded_file.getbuffer()).hexdigest()
            if (
                uploaded_hash != st.session_state.uploaded_hash
                or uploaded_file.name != st.session_state.uploaded_name
            ):
                path = _save_uploaded_file(uploaded_file)
                st.session_state.df_path = str(path)
                st.session_state.uploaded_name = uploaded_file.name
                st.session_state.uploaded_hash = uploaded_hash
                st.session_state._losses_cache = None
                st.session_state._cached_df_path = ""
                st.success(f"File ready: {uploaded_file.name}")
            else:
                st.success(f"File ready: {uploaded_file.name}")

        if st.session_state.df_path:
            st.caption(f"Temporary path: {Path(st.session_state.df_path).name}")

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input(
        "Ask for an analysis"
    )
    if prompt:
        _run_agent(prompt, confidence)
        st.rerun()


if __name__ == "__main__":
    main()

