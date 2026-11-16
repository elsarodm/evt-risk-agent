"""Quantitative tools that can be invoked by the agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_DATE_NAMES = {"date", "datetime", "timestamp", "time", "fecha"}
_PRICE_NAMES = {
    "adjustedclose",
    "adjusted_close",
    "adjclose",
    "adj_close",
    "close",
    "price",
    "precio",
    "value",
    "valor",
}


def _normalise_column_name(name: object) -> str:
    return "".join(str(name).strip().lower().split())


def _find_column(columns: pd.Index, candidates: set[str], *, exclude: set[object] | None = None,) -> object | None:
    excluded = exclude or set()
    for column in columns:
        if column in excluded:
            continue
        if _normalise_column_name(column) in candidates:
            return column
    return None


def load_and_clean_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a price history and return standardised ``date`` and ``price`` columns."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The CSV file does not contain any rows.")
    if frame.shape[1] < 2:
        raise ValueError("The CSV must contain a date column and a price column.")

    date_column = _find_column(frame.columns, _DATE_NAMES)
    if date_column is None:
        date_column = frame.columns[0]

    price_column = _find_column(
        frame.columns, _PRICE_NAMES, exclude={date_column}
    )
    if price_column is None:
        numeric_candidates = [
            column
            for column in frame.columns
            if column != date_column
            and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        if len(numeric_candidates) != 1:
            raise ValueError(
                "Could not identify a unique price column. "
                "Name it price, close, or adjusted close."
            )
        price_column = numeric_candidates[0]

    cleaned = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "price": pd.to_numeric(frame[price_column], errors="coerce"),
        }
    )
    cleaned = (
        cleaned.dropna(subset=["date", "price"])
        .loc[lambda data: data["price"] > 0]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(cleaned) < 2:
        raise ValueError("At least two valid dated, positive prices are required.")
    return cleaned


def _finite_losses(losses: Any) -> np.ndarray:
    values = np.asarray(losses, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite loss observations are required.")
    return values


def _validate_probability(value: float, name: str) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return probability


def calculate_loss_vector(df: Any) -> np.ndarray:
    """Transform positive prices into daily log-return losses."""
    if not isinstance(df, pd.DataFrame) or "price" not in df.columns:
        raise ValueError("The input data must be a DataFrame with a price column.")

    prices = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 2 or np.any(prices <= 0):
        raise ValueError("At least two finite, positive prices are required.")
    return -np.diff(np.log(prices))     # = -log(S_t / S_{t-1}) = L_t


def calculate_empirical_risk(losses: Any, alpha: float) -> dict[str, float]:
    """Calculate historical VaR and CVaR at confidence level ``alpha``."""
    values = _finite_losses(losses)
    confidence = _validate_probability(alpha, "alpha")
    var = float(np.quantile(values, confidence))
    tail = values[values >= var]
    if tail.size == 0:
        raise RuntimeError("Unable to identify observations in the VaR tail.")
    return {"var": var, "cvar": float(np.mean(tail))}


def apply_block_maxima(losses: Any, block_size: int) -> dict[str, float]:
    """Fit a GEV distribution to non-overlapping loss block maxima."""
    values = _finite_losses(losses)
    if isinstance(block_size, bool) or int(block_size) != block_size:
        raise ValueError("block_size must be a positive integer.")
    
    size = int(block_size)
    if size < 2:
        raise ValueError("block_size must be at least 2.")

    block_count = values.size // size
    if block_count < 3:
        raise ValueError("At least three complete blocks are required.")
    
    maxima = values[: block_count * size].reshape(block_count, size).max(axis=1)
    shape, location, scale = genextreme.fit(maxima)
    if not all(np.isfinite(parameter) for parameter in (shape, location, scale)):
        raise ValueError("The GEV fit returned invalid parameters.")
    if scale <= 0:
        raise ValueError("The GEV scale parameter must be positive.")
    return {
        "shape": float(shape),
        "loc": float(location),
        "scale": float(scale),
        "block_count": float(block_count),
    }


def apply_pot_method(
    losses: Any, alpha: float, threshold_quantile: float
) -> dict[str, float]:
    """Fit a GPD to threshold exceedances and calculate extreme VaR/CVaR."""
    values = _finite_losses(losses)
    confidence = _validate_probability(alpha, "alpha")
    threshold_level = _validate_probability(
        threshold_quantile, "threshold_quantile"
    )
    if confidence <= threshold_level:
        raise ValueError(
            "alpha must be greater than threshold_quantile for the POT model."
        )

    threshold = float(np.quantile(values, threshold_level))
    excesses = values[values > threshold] - threshold
    if excesses.size < 3:
        raise ValueError(
            "The POT model needs at least three observations above the threshold."
        )

    shape, _, scale = genpareto.fit(excesses, floc=0)
    shape = float(shape)
    scale = float(scale)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("The GPD fit returned invalid parameters.")
    if shape >= 1:
        raise ValueError("The fitted GPD has no finite conditional tail mean.")

    tail_probability = excesses.size / values.size      # P(L>u) = k/n
    ratio = tail_probability / (1 - confidence)         # P(L>u) / P(L>VaR) = (k/n) / (1-alpha)
    if np.isclose(shape, 0):
        var = threshold + scale * np.log(ratio)
    else:
        var = threshold + scale / shape * (ratio**shape - 1)
    cvar = var + (scale + shape * (var - threshold)) / (1 - shape)
    return {
        "var": float(var),
        "cvar": float(cvar),
        "threshold": threshold,
        "xi": shape,
        "beta": scale,
    }


def plot_tail_diagnostics(
    losses: Any, threshold: float, xi: float, beta: float
) -> str:
    """Save empirical and fitted GPD excess survival curves and return its path."""
    values = _finite_losses(losses)
    threshold_value = float(threshold)
    shape = float(xi)
    scale = float(beta)
    if not np.isfinite(threshold_value) or not np.isfinite(shape):
        raise ValueError("threshold and xi must be finite.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("beta must be a positive finite value.")

    excesses = np.sort(values[values > threshold_value] - threshold_value)
    if excesses.size == 0:
        raise ValueError("No observations exceed the supplied threshold.")

    empirical_survival = np.arange(excesses.size, 0, -1) / excesses.size
    theoretical_x = np.linspace(0, float(excesses[-1]), 200)
    theoretical_survival = genpareto.sf(
        theoretical_x, c=shape, loc=0, scale=scale
    )

    output_path = Path(__file__).resolve().parents[1] / "data" / "plot_temp.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.step(excesses, empirical_survival, where="post", label="Empirical")
    axis.plot(theoretical_x, theoretical_survival, label="Fitted GPD")
    axis.set(
        title="POT tail diagnostics",
        xlabel="Excess over threshold",
        ylabel="Survival probability",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)
