from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Pricing per 1M tokens (as of 2024)
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "gemini": {
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
        "gemini-2.5-flash-latest": {"input": 0.0, "output": 0.0},  # Free tier
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    },
    "claude": {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    },
}


@dataclass
class CostStats:
    """Cost statistics"""

    total_cost: float = 0.0
    total_tokens: int = 0
    api_calls: int = 0
    provider_costs: Dict[str, float] = field(default_factory=dict)
    provider_tokens: Dict[str, int] = field(default_factory=dict)
    provider_calls: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class CostTracker:
    """Track and calculate API costs"""

    def __init__(self):
        self.stats = CostStats()
        self.history = []

    def calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """
        Calculate cost for API call

        Args:
            provider: Provider name (openai, gemini, claude)
            model: Model name
            tokens: Total tokens used

        Returns:
            Cost in USD
        """
        provider_key = provider.value if hasattr(provider, "value") else provider

        if provider_key not in PRICING:
            logger.warning("Unknown provider: %s, assuming $0 cost", provider_key)
            return 0.0

        if model not in PRICING[provider_key]:
            logger.warning(
                "Unknown model: %s for provider %s, using default pricing",
                model,
                provider_key,
            )
            # Use first model's pricing as default
            model = list(PRICING[provider_key].keys())[0]

        pricing = PRICING[provider_key][model]

        # Simplified cost calculation (assume 50/50 input/output split)
        # In production, track input/output separately
        avg_price = (pricing["input"] + pricing["output"]) / 2
        cost = (tokens / 1_000_000) * avg_price

        # Update statistics
        self._update_stats(provider_key, tokens, cost)

        return cost

    def _update_stats(self, provider: str, tokens: int, cost: float):
        """Update internal statistics"""
        self.stats.total_cost += cost
        self.stats.total_tokens += tokens
        self.stats.api_calls += 1

        # Provider-specific stats
        if provider not in self.stats.provider_costs:
            self.stats.provider_costs[provider] = 0.0
            self.stats.provider_tokens[provider] = 0
            self.stats.provider_calls[provider] = 0

        self.stats.provider_costs[provider] += cost
        self.stats.provider_tokens[provider] += tokens
        self.stats.provider_calls[provider] += 1

        # Add to history
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "tokens": tokens,
                "cost": cost,
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        runtime = (datetime.now() - self.stats.start_time).total_seconds()

        return {
            "total_cost": round(self.stats.total_cost, 4),
            "total_tokens": self.stats.total_tokens,
            "api_calls": self.stats.api_calls,
            "runtime_seconds": runtime,
            "cost_per_call": (
                round(self.stats.total_cost / self.stats.api_calls, 4)
                if self.stats.api_calls > 0
                else 0
            ),
            "tokens_per_call": (
                self.stats.total_tokens // self.stats.api_calls
                if self.stats.api_calls > 0
                else 0
            ),
            "providers": {
                provider: {
                    "cost": round(cost, 4),
                    "tokens": self.stats.provider_tokens[provider],
                    "calls": self.stats.provider_calls[provider],
                }
                for provider, cost in self.stats.provider_costs.items()
            },
        }

    def export_history(self, filepath: str):
        """Export history to JSON file"""
        with open(filepath, "w") as f:
            json.dump(
                {"stats": self.get_stats(), "history": self.history}, f, indent=2
            )

        logger.info("Cost history exported to %s", filepath)

    def check_budget(self, budget: float) -> Dict[str, Any]:
        """Check if budget threshold is exceeded"""
        used_percentage = (self.stats.total_cost / budget) * 100

        return {
            "budget": budget,
            "used": round(self.stats.total_cost, 4),
            "remaining": round(budget - self.stats.total_cost, 4),
            "used_percentage": round(used_percentage, 2),
            "exceeded": self.stats.total_cost > budget,
        }

    def reset(self):
        """Reset all statistics"""
        self.stats = CostStats()
        self.history = []
        logger.info("Cost tracker reset")
