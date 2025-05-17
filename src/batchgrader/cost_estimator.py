"""
Cost Estimator for OpenAI API batch pricing.
"""

import os
from typing import Dict, Optional, Tuple


class CostEstimator:
    """
    Estimates API costs based on the pricing data in docs/pricing.csv.

    Note:
        The pricing.csv file should use plain numbers (no $ or anything else) for prices. Example:
        gpt-4.1-2025-04-14,1.00,4.00

    Usage:
        estimator = CostEstimator()
        cost = estimator.estimate_cost('gpt-4o-2024-08-06', 1200000, 800000)
        print(f"Estimated cost: ${cost:.4f}")
    """

    _pricing: Optional[Dict[str, Tuple[float, float]]] = None
    _csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "docs", "pricing.csv")

    @classmethod
    def _load_pricing(cls) -> None:  # type: ignore[no-untyped-def]
        """Load pricing data from the CSV file."""
        import csv

        cls._pricing = {}
        try:
            with open(cls._csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and not row[0].startswith("#"):
                        model = row[0].strip()
                        input_price = float(row[1].strip())
                        output_price = float(row[2].strip())
                        cls._pricing[model] = (input_price, output_price)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pricing file not found at {cls._csv_path}") from None
        except Exception as e:
            raise RuntimeError( 
                f"Error loading pricing data: {str(e)}") from e

    @classmethod
    def estimate_cost(cls, model: str, n_input_tokens: int,
                      n_output_tokens: int) -> float:
        """Estimate the cost for a given model and token counts.

        Args:
            model: Model name as in pricing.csv
            n_input_tokens: Number of input tokens
            n_output_tokens: Number of output tokens
        Returns:
            Estimated cost in USD (float)
        Raises:
            ValueError: If the model is not found in the pricing table.
        """
        if cls._pricing is None:
            cls._load_pricing()
        if model not in cls._pricing:
            raise ValueError(f"Model '{model}' not found in pricing table.")
        input_price, output_price = cls._pricing[model]
        return (n_input_tokens / 1_000_000) * input_price + (
            n_output_tokens / 1_000_000) * output_price

    @classmethod
    def available_models(cls):
        """
        Returns a list of available model names from the pricing table.
        """
        if cls._pricing is None or not isinstance(cls._pricing, dict):
            cls._load_pricing()
        return list(cls._pricing.keys()) if cls._pricing else []

    @classmethod
    def print_available_models(cls):
        """
        Prints a list of available model names from the pricing table.
        """
        models = cls.available_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"- {model}")
        else:
            print("No models found in pricing table.")


if __name__ == "__main__":  # pragma: no cover
    estimator = CostEstimator()
    cost = estimator.estimate_cost("gpt-4o-2024-08-06", 1200000, 800000)
    print(f"Estimated cost: ${cost:.4f}")
    estimator.print_available_models()
