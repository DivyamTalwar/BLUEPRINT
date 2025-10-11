import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

from openai import OpenAI
import google.generativeai as genai

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


class SimpleLLMRouter:
    """Simplified router using OpenRouter + Gemini"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with available API keys"""
        self.config = config
        self.total_cost = 0.0
        self.api_calls = 0

        # Check available keys
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        gemini_key = os.getenv("GOOGLE_API_KEY")

        if not openrouter_key and not gemini_key:
            raise ValueError("Need at least OPENROUTER_API_KEY or GOOGLE_API_KEY in .env")

        # Initialize OpenRouter (for GPT-4, Claude, etc.)
        if openrouter_key:
            self.openrouter = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.has_openrouter = True
            # Use GPT-4o-mini through OpenRouter (cheaper)
            self.openrouter_model = "openai/gpt-4o-mini"
            logger.info("OpenRouter initialized: %s", self.openrouter_model)
        else:
            self.has_openrouter = False

        # Initialize Gemini (free!)
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = "gemini-2.0-flash-exp"
            self.has_gemini = True
            logger.info("Gemini initialized: %s", self.gemini_model)
        else:
            self.has_gemini = False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
        prefer_free: bool = True,
    ) -> LLMResponse:
        """
        Generate response

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            json_mode: Force JSON output
            prefer_free: Use Gemini first (free)

        Returns:
            LLM response
        """
        # Try Gemini first if available and prefer_free
        if self.has_gemini and prefer_free:
            try:
                return self._call_gemini(prompt, temperature, max_tokens, json_mode)
            except Exception as e:
                logger.warning("Gemini failed: %s, trying OpenRouter", str(e))
                if not self.has_openrouter:
                    raise

        # Fall back to OpenRouter
        if self.has_openrouter:
            return self._call_openrouter(prompt, temperature, max_tokens, json_mode)

        # Last resort: Gemini
        if self.has_gemini:
            return self._call_gemini(prompt, temperature, max_tokens, json_mode)

        raise ValueError("No LLM providers available")

    def _call_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Call Gemini (FREE)"""
        start_time = time.time()

        try:
            model = genai.GenerativeModel(self.gemini_model)

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            response = model.generate_content(prompt, generation_config=generation_config)

            latency = time.time() - start_time
            tokens = len(response.text.split()) * 1.3  # Estimate
            cost = 0.0  # FREE!

            self.api_calls += 1

            logger.debug("Gemini: tokens~%d, cost=$0 (FREE), latency=%.2fs", tokens, latency)

            return LLMResponse(
                content=response.text,
                provider="gemini",
                model=self.gemini_model,
                tokens_used=int(tokens),
                cost=cost,
                latency=latency,
                metadata={"finish_reason": "stop"},
            )

        except Exception as e:
            logger.error("Gemini error: %s", str(e))
            raise

    def _call_openrouter(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Call OpenRouter"""
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]

            if json_mode:
                # Add JSON instruction
                messages[0]["content"] = f"{prompt}\n\nRespond with valid JSON only."

            response = self.openrouter.chat.completions.create(
                model=self.openrouter_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency = time.time() - start_time
            tokens = response.usage.total_tokens if response.usage else 1000

            # OpenRouter pricing for gpt-4o-mini: ~$0.15/1M input, $0.60/1M output
            # Estimate: ~$0.40/1M average
            cost = (tokens / 1_000_000) * 0.40

            self.total_cost += cost
            self.api_calls += 1

            logger.debug("OpenRouter: tokens=%d, cost=$%.4f, latency=%.2fs", tokens, cost, latency)

            return LLMResponse(
                content=response.choices[0].message.content,
                provider="openrouter",
                model=self.openrouter_model,
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error("OpenRouter error: %s", str(e))
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_cost": round(self.total_cost, 4),
            "api_calls": self.api_calls,
            "cost_per_call": round(self.total_cost / self.api_calls, 4) if self.api_calls > 0 else 0,
            "providers": {
                "openrouter": self.has_openrouter,
                "gemini": self.has_gemini,
            }
        }
