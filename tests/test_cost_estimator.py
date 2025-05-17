"""
Unit tests for the cost_estimator module.
"""

from unittest.mock import mock_open, patch

import pytest

from batchgrader.cost_estimator import CostEstimator


@pytest.fixture
def mock_pricing_csv():
    """Mock pricing CSV data for tests."""
    csv_content = ("Model,Input,Output\n"
                   "gpt-4o-mini-2024-07-18,0.15,0.60\n"
                   "gpt-4o-2024-08-06,1.00,3.00\n"
                   "gpt-3.5-turbo,0.50,1.50\n")
    return csv_content


@pytest.fixture
def setup_pricing_csv(mock_pricing_csv, tmp_path):
    """Setup a temporary pricing CSV file for tests."""
    pricing_file = tmp_path / "pricing.csv"
    pricing_file.write_text(mock_pricing_csv)
    return str(pricing_file)


def test_load_pricing(mock_pricing_csv):
    """Test loading of pricing data from CSV."""
    with patch("builtins.open", mock_open(read_data=mock_pricing_csv)):
        CostEstimator._pricing = None  # Reset the class variable
        CostEstimator._load_pricing()

        # Check that pricing data was loaded correctly
        assert "gpt-4o-mini-2024-07-18" in CostEstimator._pricing
        assert CostEstimator._pricing[
            "gpt-4o-mini-2024-07-18"] == pytest.approx((0.15, 0.60))
        assert "gpt-4o-2024-08-06" in CostEstimator._pricing
        assert CostEstimator._pricing["gpt-4o-2024-08-06"] == pytest.approx(
            (1.00, 3.00))
        assert "gpt-3.5-turbo" in CostEstimator._pricing
        assert CostEstimator._pricing["gpt-3.5-turbo"] == pytest.approx(
            (0.50, 1.50))


def test_estimate_cost_with_valid_model(mock_pricing_csv):
    """Test cost estimation with a valid model."""
    with patch("builtins.open", mock_open(read_data=mock_pricing_csv)):
        CostEstimator._pricing = None  # Reset the class variable

        # 1M input tokens, 0.5M output tokens for gpt-4o-2024-08-06
        cost = CostEstimator.estimate_cost("gpt-4o-2024-08-06", 1_000_000,
                                           500_000)
        # Expected: (1.0 * 1 + 3.0 * 0.5) = 2.5
        assert cost == pytest.approx(2.5)

        # 0.5M input tokens, 0.2M output tokens for gpt-4o-mini-2024-07-18
        cost = CostEstimator.estimate_cost("gpt-4o-mini-2024-07-18", 500_000,
                                           200_000)
        # Expected: (0.15 * 0.5 + 0.6 * 0.2) = 0.195
        assert cost == pytest.approx(0.195)

        # Test with different token counts
        cost = CostEstimator.estimate_cost("gpt-3.5-turbo", 2_000_000,
                                           1_000_000)
        # Expected: (0.5 * 2 + 1.5 * 1) = 2.5
        assert cost == pytest.approx(2.5)


def test_estimate_cost_with_invalid_model(mock_pricing_csv):
    """Test cost estimation with an invalid model."""
    with patch("builtins.open", mock_open(read_data=mock_pricing_csv)):
        CostEstimator._pricing = None  # Reset the class variable

        # Should raise ValueError for non-existent model
        with pytest.raises(ValueError) as excinfo:
            CostEstimator.estimate_cost("nonexistent-model", 1000, 500)

        assert "not found in pricing table" in str(excinfo.value)


def test_missing_pricing_file():
    """Test behavior when pricing file is missing."""
    with patch("builtins.open", side_effect=FileNotFoundError()):
        CostEstimator._pricing = None  # Reset the class variable

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            CostEstimator.estimate_cost("gpt-4o-2024-08-06", 1000, 500)


def test_csv_path_exists():
    """Test that the CSV path is correctly defined."""
    path = CostEstimator._csv_path.replace("\\",
                                           "/")  # Normalize path separators
    assert path.endswith("docs/pricing.csv")
    assert "docs" in path
    assert "pricing.csv" in path
