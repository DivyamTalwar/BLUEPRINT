import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

from openai import OpenAI

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    provider: str
    model: str
    tokens_used: int
    cost: float
    latency: float
    metadata: Dict[str, Any]


class FinalLLMRouter:
    """
    Production LLM Router - Claude Only

    Strategy:
    - Claude 3.5 Sonnet: Simple tasks (feature selection, basic queries)
    - Claude 3.7 Sonnet: Complex tasks (code generation, architecture design)

    NO GEMINI - It's safety filters are too aggressive and unusable
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with OpenRouter API key"""
        self.config = config
        self.total_cost = 0.0
        self.api_calls = 0

        # Check required keys
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if not openrouter_key:
            raise ValueError("Need OPENROUTER_API_KEY in .env")

        # Initialize OpenRouter for Claude
        self.openrouter = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Model selection
        self.claude_simple = "anthropic/claude-3.5-sonnet"  # Simple tasks
        self.claude_complex = "anthropic/claude-3.7-sonnet"  # Complex tasks

        logger.info("[OK] OpenRouter initialized: %s (simple), %s (complex)",
                   self.claude_simple, self.claude_complex)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
        prefer_claude: bool = False,  # Deprecated but kept for compatibility
        complexity: str = "auto",  # "simple", "complex", or "auto"
    ) -> LLMResponse:
        """
        Generate response using Claude

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            json_mode: Force JSON output
            prefer_claude: Deprecated (always uses Claude now)
            complexity: Task complexity ("simple", "complex", or "auto")

        Returns:
            LLM response
        """
        # Determine which model to use based on complexity
        if complexity == "auto":
            complexity = self._detect_complexity(prompt, max_tokens)

        if complexity == "complex":
            model = self.claude_complex
            logger.debug("Using Claude 3.7 Sonnet (complex task)")
        else:
            model = self.claude_simple
            logger.debug("Using Claude 3.5 Sonnet (simple task)")

        return self._call_claude(model, prompt, temperature, max_tokens, json_mode)

    def _detect_complexity(self, prompt: str, max_tokens: int) -> str:
        """
        Detect task complexity based on prompt characteristics

        Complex tasks:
        - Code generation (>1000 tokens)
        - Architecture design
        - Large context (>2000 tokens)
        - Multiple requirements

        Simple tasks:
        - Feature selection
        - Simple queries
        - Small responses (<1000 tokens)
        """
        # Check token budget (complex tasks need more tokens)
        if max_tokens > 2000:
            return "complex"

        # Check prompt length (longer prompts = complex tasks)
        if len(prompt) > 2000:
            return "complex"

        # Check for code generation keywords
        code_keywords = [
            "generate code",
            "implement",
            "write function",
            "create class",
            "build",
            "design architecture",
            "refactor",
            "optimize",
        ]

        prompt_lower = prompt.lower()
        for keyword in code_keywords:
            if keyword in prompt_lower:
                return "complex"

        # Check for multi-step reasoning
        multi_step_keywords = [
            "step by step",
            "analyze and",
            "design and implement",
            "plan and",
            "multiple",
            "several",
        ]

        for keyword in multi_step_keywords:
            if keyword in prompt_lower:
                return "complex"

        # Default to simple
        return "simple"

    def _call_claude(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Call Claude via OpenRouter"""
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]

            if json_mode:
                messages[0]["content"] = f"{prompt}\n\nIMPORTANT: Respond with ONLY valid JSON. No explanations, no markdown, no code fences. Start with {{ and end with }}."

            response = self.openrouter.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency = time.time() - start_time
            tokens = response.usage.total_tokens if response.usage else 1000

            # OpenRouter pricing:
            # Claude 3.5 Sonnet: Input $3/1M, Output $15/1M (avg ~$9/1M)
            # Claude 3.7 Sonnet: Input $3/1M, Output $15/1M (avg ~$9/1M) - same pricing
            cost = (tokens / 1_000_000) * 9.0

            self.total_cost += cost
            self.api_calls += 1

            model_name = "3.7" if "3.7" in model else "3.5"
            logger.debug("Claude %s: tokens=%d, cost=$%.4f, latency=%.2fs",
                        model_name, tokens, cost, latency)

            content = response.choices[0].message.content

            # Extract JSON from markdown code fences if present
            if json_mode and "```" in content:
                # Extract JSON from ```json or ``` code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

            return LLMResponse(
                content=content,
                provider="claude",
                model=model,
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error("Claude (OpenRouter) error: %s", str(e))
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_cost": round(self.total_cost, 4),
            "api_calls": self.api_calls,
            "cost_per_call": round(self.total_cost / self.api_calls, 4) if self.api_calls > 0 else 0,
            "providers": {
                "claude_3.5_sonnet": True,
                "claude_3.7_sonnet": True,
            },
            "total_tokens": 0,  # For compatibility
        }
