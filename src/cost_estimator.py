"""
Cost Estimator for OpenAI API batch pricing.
"""
import os
import csv
from pathlib import Path
from config_loader import load_config
from typing import Dict, Tuple

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
    _pricing: Dict[str, Tuple[float, float]] = None
    _csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'pricing.csv')

    @classmethod
    def _load_pricing(cls):
        pricing = {}
        with open(cls._csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model = row['Model']
                input_price = float(row['Input'])
                output_price = float(row['Output'])
                pricing[model] = (input_price, output_price)
        cls._pricing = pricing

    @classmethod
    def estimate_cost(cls, model: str, n_input_tokens: int, n_output_tokens: int) -> float:
        """
        Estimate the cost for a given model and token counts.

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
        cost = (n_input_tokens / 1_000_000) * input_price + (n_output_tokens / 1_000_000) * output_price
        return cost

if __name__ == "__main__": # this is just here for testing
    estimator = CostEstimator()
    cost = estimator.estimate_cost("gpt-4o-2024-08-06", 1200000, 800000)
    print(f"Estimated cost: ${cost:.4f}") 