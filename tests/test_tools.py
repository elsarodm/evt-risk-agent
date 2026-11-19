from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent import tools


def test_load_and_clean_csv_normalises_dates_prices_and_duplicates(tmp_path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Fecha": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01", "invalid", "2024-01-04"],
            "Cierre": [110, 100, -1, 105, 50, 120],
        }
    ).to_csv(csv_path, index=False)

    cleaned = tools.load_and_clean_csv(csv_path)

    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-03",
        "2024-01-04",
    ]
    assert cleaned["price"].tolist() == [105, 110, 120]


def test_calculate_loss_vector_returns_logarithmic_losses():
    prices = pd.DataFrame({"price": [100.0, 110.0, 99.0]})

    losses = tools.calculate_loss_vector(prices)
    expected = np.array([-np.log(1.1), -np.log(0.9)])

    np.testing.assert_allclose(losses, expected)
    assert losses[0] != 0
    assert losses[1] != 0


def test_calculate_empirical_risk_returns_var_and_avar():
    result = tools.calculate_empirical_risk([0.0, 0.1, 0.2, 0.3], alpha=0.75)

    assert result["var"] == pytest.approx(0.225)
    assert result["avar"] == pytest.approx(0.3)


def test_calculate_empirical_risk_uses_95th_percentile_and_conditional_mean():
    result = tools.calculate_empirical_risk(np.arange(1, 101), alpha=0.95)

    assert result["var"] == pytest.approx(95.05)
    assert result["avar"] == pytest.approx(98.0)


def test_apply_pot_method_rejects_threshold_quantile_at_or_above_alpha():
    losses = np.linspace(0.01, 0.10, 10)

    with pytest.raises(ValueError, match="strictly below alpha"):
        tools.apply_pot_method(
            losses,
            alpha=0.95,
            threshold_quantile=0.99,
        )


def test_apply_pot_method_rejects_absolute_threshold_at_or_above_alpha():
    losses = np.linspace(0.01, 0.10, 10)

    with pytest.raises(ValueError, match="strictly below alpha"):
        tools.apply_pot_method(
            losses, 
            alpha=0.90, 
            threshold=0.095)


def test_apply_block_maxima_distributes_requested_blocks():
    result = tools.apply_block_maxima(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        block_count=3,
        alpha=0.99,
    )

    assert result["block_count"] == 3.0
    assert result["block_size"] == 3.0
    assert len(result["block_maxima"]) == 3
    assert result["return_period"] == pytest.approx(100.0)


def test_apply_block_maxima_converts_scipy_shape_to_conventional_xi(monkeypatch):
    monkeypatch.setattr(
        tools.genextreme,
        "fit",
        lambda maxima: (0.25, 2.0, 3.0),
    )

    result = tools.apply_block_maxima(
        np.arange(1.0, 10.0),
        block_count=3,
        alpha=0.99,
    )

    assert result["xi"] == pytest.approx(-0.25)
    assert result["mu"] == pytest.approx(2.0)
    assert result["sigma"] == pytest.approx(3.0)
    assert {"xi", "mu", "sigma", "return_level", "return_period"} <= result.keys()


@pytest.mark.parametrize(
    "losses",
    [
        [],
        [0.1],
        [0.1, np.nan],
    ],
)
def test_calculate_empirical_risk_requires_two_finite_losses(losses):
    with pytest.raises(ValueError, match="two finite"):
        tools.calculate_empirical_risk(losses, alpha=0.99)
