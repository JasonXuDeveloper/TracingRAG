"""LLM client for RAG generation using OpenRouter or OpenAI-compatible APIs"""

import os
from typing import Any

import httpx

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest, LLMResponse


class LLMClient:
    """Client for interacting with LLMs via OpenRouter or OpenAI-compatible APIs"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "anthropic/claude-3.5-sonnet",
        timeout: float = 120.0,
    ):
        """
        Initialize LLM client

        Args:
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Base URL for API (OpenRouter by default)
            default_model: Default model to use
            timeout: Request timeout in seconds
        """
        # Try api_key parameter, then settings, then OS env var
        self.api_key = api_key or settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY in .env file, environment variable, "
                "or pass api_key parameter"
            )

        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

        # HTTP client for async requests
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from LLM

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLM response with generated content
        """
        # Build messages for chat completion
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"{request.context}\n\nQuery: {request.user_message}"},
        ]

        # Build request payload
        payload = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Enable structured output if requested
        if request.json_schema:
            # Use JSON schema for strict structured output (OpenRouter format)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": request.json_schema,
            }
        elif request.json_mode:
            # Fallback to simple JSON mode (less strict)
            payload["response_format"] = {"type": "json_object"}

        # Add OpenRouter-specific headers if using OpenRouter
        headers = {}
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = request.metadata.get(
                "referer", "https://github.com/JasonXuDeveloper/TracingRAG"
            )
            headers["X-Title"] = request.metadata.get("title", "TracingRAG")

        # Make request
        response = await self.client.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )

        # Enhanced error handling for debugging
        if response.status_code == 429:
            error_body = response.text
            raise Exception(f"Rate limit (429) - Response: {error_body}")

        if response.status_code == 400:
            error_body = response.text
            # Check if it's a JSON schema issue
            if request.json_schema and (
                "json_schema" in error_body.lower() or "response_format" in error_body.lower()
            ):
                raise Exception(
                    f"Model {request.model} does not support JSON schema structured output. "
                    f"Error: {error_body}"
                )
            raise Exception(f"Bad request (400) - Model: {request.model} - Response: {error_body}")

        response.raise_for_status()

        # Parse response
        data = response.json()

        # Extract content and metadata
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")

        # Get token usage if available
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", request.model),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            metadata={
                "usage": usage,
                "response_id": data.get("id"),
                "created": data.get("created"),
            },
        )

    async def generate_streaming(self, request: LLMRequest) -> Any:  # AsyncGenerator[str, None]
        """
        Generate response with streaming (for future use)

        Args:
            request: LLM request with prompt and parameters

        Yields:
            Chunks of generated text
        """
        # Build messages
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"{request.context}\n\nQuery: {request.user_message}"},
        ]

        # Build request payload
        payload = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        # Add OpenRouter-specific headers if using OpenRouter
        headers = {}
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = request.metadata.get(
                "referer", "https://github.com/TracingRAG"
            )
            headers["X-Title"] = request.metadata.get("title", "TracingRAG")

        # Make streaming request
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Singleton instance
_llm_client: LLMClient | None = None


def get_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
) -> LLMClient:
    """
    Get or create LLM client singleton

    Args:
        api_key: API key (defaults to settings.openrouter_api_key)
        base_url: Base URL for API (defaults to settings.openrouter_base_url)
        default_model: Default model to use (defaults to settings.default_llm_model)

    Returns:
        LLM client instance
    """
    global _llm_client

    if _llm_client is None:
        _llm_client = LLMClient(
            api_key=api_key,
            base_url=base_url or settings.openrouter_base_url,
            default_model=default_model or settings.default_llm_model,
        )

    return _llm_client


async def close_llm_client():
    """Close the global LLM client"""
    global _llm_client

    if _llm_client is not None:
        await _llm_client.close()
        _llm_client = None
