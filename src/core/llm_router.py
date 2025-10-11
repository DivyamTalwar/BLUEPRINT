import os
import random
import time
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
from enum import Enum
import json
import logging

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

from src.utils.cost_tracker import CostTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    provider: ModelProvider
    model: str
    tokens_used: int
    cost: float
    latency: float
    metadata: Dict[str, Any]


class LLMRouter:
    """
    Unified LLM client with random routing and fallback support
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_tracker = CostTracker()

        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Model configurations
        self.models = {
            ModelProvider.OPENAI: config["api"]["openai"]["model"],
            ModelProvider.GEMINI: config["api"]["gemini"]["model"],
            ModelProvider.CLAUDE: config["api"]["claude"]["model"],
        }

        # Router settings
        self.random_distribution = config["api"]["router"]["random_distribution"]
        self.fallback_enabled = config["api"]["router"]["fallback_enabled"]
        self.retry_attempts = config["api"]["router"]["retry_attempts"]

        logger.info("LLMRouter initialized with models: %s", self.models)

    def _select_provider(
        self, force_provider: Optional[ModelProvider] = None
    ) -> ModelProvider:
        """Select LLM provider (random or forced)"""
        if force_provider:
            return force_provider

        if self.random_distribution:
            return random.choice(list(ModelProvider))

        # Default to OpenAI
        return ModelProvider.OPENAI

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _call_openai(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000, json_mode: bool = False
    ) -> LLMResponse:
        """Call OpenAI API"""
        start_time = time.time()

        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}

            response = self.openai_client.chat.completions.create(
                model=self.models[ModelProvider.OPENAI],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            latency = time.time() - start_time
            tokens = response.usage.total_tokens
            cost = self.cost_tracker.calculate_cost(
                ModelProvider.OPENAI, self.models[ModelProvider.OPENAI], tokens
            )

            logger.debug(
                "OpenAI API call completed: tokens=%d, cost=$%.4f, latency=%.2fs",
                tokens,
                cost,
                latency,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                provider=ModelProvider.OPENAI,
                model=self.models[ModelProvider.OPENAI],
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error("OpenAI API error: %s", str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _call_gemini(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000, json_mode: bool = False
    ) -> LLMResponse:
        """Call Gemini API"""
        start_time = time.time()

        try:
            model = genai.GenerativeModel(self.models[ModelProvider.GEMINI])

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            response = model.generate_content(
                prompt, generation_config=generation_config
            )

            latency = time.time() - start_time
            # Gemini doesn't provide token count directly, estimate
            tokens = len(response.text.split()) * 1.3  # rough estimate
            cost = self.cost_tracker.calculate_cost(
                ModelProvider.GEMINI, self.models[ModelProvider.GEMINI], int(tokens)
            )

            logger.debug(
                "Gemini API call completed: tokens~%d, cost=$%.4f, latency=%.2fs",
                tokens,
                cost,
                latency,
            )

            return LLMResponse(
                content=response.text,
                provider=ModelProvider.GEMINI,
                model=self.models[ModelProvider.GEMINI],
                tokens_used=int(tokens),
                cost=cost,
                latency=latency,
                metadata={"finish_reason": "stop"},
            )

        except Exception as e:
            logger.error("Gemini API error: %s", str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _call_claude(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000, json_mode: bool = False
    ) -> LLMResponse:
        """Call Claude API"""
        start_time = time.time()

        try:
            system_message = ""
            if json_mode:
                system_message = "You must respond with valid JSON only. No other text."

            response = self.anthropic_client.messages.create(
                model=self.models[ModelProvider.CLAUDE],
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=[{"role": "user", "content": prompt}],
            )

            latency = time.time() - start_time
            tokens = response.usage.input_tokens + response.usage.output_tokens
            cost = self.cost_tracker.calculate_cost(
                ModelProvider.CLAUDE, self.models[ModelProvider.CLAUDE], tokens
            )

            logger.debug(
                "Claude API call completed: tokens=%d, cost=$%.4f, latency=%.2fs",
                tokens,
                cost,
                latency,
            )

            return LLMResponse(
                content=response.content[0].text,
                provider=ModelProvider.CLAUDE,
                model=self.models[ModelProvider.CLAUDE],
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            logger.error("Claude API error: %s", str(e))
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
        force_provider: Optional[ModelProvider] = None,
    ) -> LLMResponse:
        """
        Generate response using selected LLM provider with fallback

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to enforce JSON output
            force_provider: Force specific provider (optional)

        Returns:
            LLMResponse object with generated content
        """
        provider = self._select_provider(force_provider)

        providers_to_try = [provider]
        if self.fallback_enabled:
            # Add other providers as fallback
            all_providers = list(ModelProvider)
            fallbacks = [p for p in all_providers if p != provider]
            providers_to_try.extend(fallbacks)

        last_exception = None

        for attempt_provider in providers_to_try:
            try:
                logger.info("Attempting LLM call with provider: %s", attempt_provider)

                if attempt_provider == ModelProvider.OPENAI:
                    return self._call_openai(prompt, temperature, max_tokens, json_mode)
                elif attempt_provider == ModelProvider.GEMINI:
                    return self._call_gemini(prompt, temperature, max_tokens, json_mode)
                elif attempt_provider == ModelProvider.CLAUDE:
                    return self._call_claude(prompt, temperature, max_tokens, json_mode)

            except Exception as e:
                last_exception = e
                logger.warning(
                    "Provider %s failed: %s. Trying fallback...",
                    attempt_provider,
                    str(e),
                )
                continue

        # All providers failed
        logger.error("All LLM providers failed. Last error: %s", str(last_exception))
        raise Exception(f"All LLM providers failed: {str(last_exception)}")

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            json_mode: Whether to enforce JSON output

        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, temperature, max_tokens, json_mode)
            responses.append(response)
        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.cost_tracker.get_stats()

    def validate_json_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Validate and parse JSON response"""
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON response: %s", str(e))
            raise ValueError(f"Invalid JSON response: {str(e)}")
