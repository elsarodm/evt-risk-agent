"""Quantitative tools that can be invoked by the agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_and_clean_csv(csv_path: str | Path) -> Any:
    """Load and clean a historical CSV file.

    The statistical implementation will be added to this module.
    """
    raise NotImplementedError


def calculate_loss_vector(df: Any) -> Any:
    """Transform prices into log returns and losses."""
    raise NotImplementedError


def calculate_empirical_risk(
    losses: Any, alpha: float
) -> dict[str, float]:
    """Calculate empirical VaR and CVaR."""
    raise NotImplementedError


def apply_pot_method(
    losses: Any, alpha: float, threshold_quantile: float
) -> dict[str, float]:
    """Fit a GPD to exceedances and calculate extreme quantiles."""
    raise NotImplementedError


def apply_block_maxima(
    losses: Any, block_size: int
) -> dict[str, float]:
    """Fit a GEV distribution to block maxima."""
    raise NotImplementedError


def plot_tail_diagnostics(
    losses: Any, threshold: float, xi: float, beta: float
) -> str:
    """Generate and return the path to a tail diagnostic plot."""
    raise NotImplementedError
