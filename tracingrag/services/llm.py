"""LLM client for RAG generation using OpenRouter or OpenAI-compatible APIs"""

import asyncio
import os
import random
from typing import Any

import httpx

from tracingrag.config import settings
from tracingrag.core.models.rag import LLMRequest, LLMResponse
from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


def _add_additional_properties_false(schema: dict) -> dict:
    """
    Recursively add 'additionalProperties: false' and ensure all properties are required

    OpenAI strict mode requires:
    1. additionalProperties: false for all object types (including nested)
    2. All properties must be in the 'required' array (no optional fields)
    3. $ref cannot have sibling keywords (like 'description', 'title', etc.)
    """
    if not isinstance(schema, dict):
        return schema

    # Make a copy to avoid modifying the original
    result = schema.copy()

    # If this schema has a $ref, remove all other keywords (OpenAI strict mode requirement)
    if "$ref" in result:
        # Keep only $ref, remove everything else (description, title, etc.)
        return {"$ref": result["$ref"]}

    # Remove type: None if present (invalid for OpenAI)
    if "type" in result and result["type"] is None:
        del result["type"]

    # If this is an object type, ensure additionalProperties is false
    if result.get("type") == "object":
        result["additionalProperties"] = False

        # OpenAI strict mode: all properties must be required
        if "properties" in result:
            # Add all property keys to required array
            all_properties = list(result["properties"].keys())
            result["required"] = all_properties

    # Recursively process nested schemas
    if "properties" in result:
        result["properties"] = {
            key: _add_additional_properties_false(value)
            for key, value in result["properties"].items()
        }

    if "items" in result:
        result["items"] = _add_additional_properties_false(result["items"])

    if "anyOf" in result:
        result["anyOf"] = [_add_additional_properties_false(item) for item in result["anyOf"]]

    if "allOf" in result:
        result["allOf"] = [_add_additional_properties_false(item) for item in result["allOf"]]

    if "oneOf" in result:
        result["oneOf"] = [_add_additional_properties_false(item) for item in result["oneOf"]]

    # Handle definitions/components (Pydantic may put nested schemas here)
    if "$defs" in result:
        result["$defs"] = {
            key: _add_additional_properties_false(value) for key, value in result["$defs"].items()
        }

    return result


def format_schema_for_openrouter(model: str, schema_name: str, base_schema: dict) -> dict:
    """
    Format JSON schema for different model providers via OpenRouter

    OpenAI models require strict mode with additionalProperties: false on all objects,
    while Google/Anthropic models use simpler format.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini", "google/gemini-2.5-flash")
        schema_name: Name of the schema (e.g., "query_analysis")
        base_schema: Pydantic model's .model_json_schema() output

    Returns:
        Formatted response_format dict for OpenRouter API
    """
    # OpenAI models use strict mode and require additionalProperties: false on all objects
    if model.startswith("openai/"):
        import json

        # Log the original schema
        logger.debug(
            f"[OpenAI Schema] Original base_schema for '{schema_name}': {json.dumps(base_schema, indent=2)}"
        )

        # Recursively add additionalProperties: false to all objects
        strict_schema = _add_additional_properties_false(base_schema)

        # Log after processing
        logger.debug(
            f"[OpenAI Schema] After _add_additional_properties_false: {json.dumps(strict_schema, indent=2)}"
        )

        # Ensure the top-level schema has type: object (OpenAI requirement)
        # Remove None values from type field if present
        if "type" in strict_schema and strict_schema["type"] is None:
            logger.warning(
                f"[OpenAI Schema] Found type: None in schema '{schema_name}', removing it"
            )
            del strict_schema["type"]

        # If schema has type: object but no properties, it's invalid
        if strict_schema.get("type") == "object" and "properties" not in strict_schema:
            # This shouldn't happen with Pydantic schemas, but log it for debugging
            logger.error(
                f"[OpenAI Schema] Schema '{schema_name}' has type: object but no properties! "
                f"Schema: {json.dumps(strict_schema, indent=2)}"
            )

        # Ensure object schemas have additionalProperties: false
        if strict_schema.get("type") == "object" and "additionalProperties" not in strict_schema:
            strict_schema["additionalProperties"] = False

        # Final schema to send
        final_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": strict_schema,
            },
        }

        logger.debug(
            f"[OpenAI Schema] Final formatted schema for '{schema_name}': {json.dumps(final_schema, indent=2)}"
        )

        return final_schema

    # Google/Anthropic/other models use simpler format
    else:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": base_schema,
            },
        }


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
        Generate response from LLM with retry logic and fallback model support

        Strategy:
        1. Try primary model (request.model) with llm_max_retries attempts
        2. If primary fails, try fallback model (settings.fallback_llm_model) with fallback_llm_max_retries attempts
        3. Only raise exception if both primary and fallback fail

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLM response with generated content

        Raises:
            Exception: If all retry attempts (primary + fallback) fail
        """
        primary_model = request.model or self.default_model
        fallback_model = settings.fallback_llm_model

        # Try primary model first
        try:
            logger.debug(f"Attempting with primary model: {primary_model}")
            return await self._generate_with_model(request, primary_model, settings.llm_max_retries)
        except Exception as primary_error:
            # Primary model failed after all retries, try fallback
            logger.warning(
                f"Primary model {primary_model} failed after {settings.llm_max_retries} retries. "
                f"Switching to fallback model: {fallback_model}"
            )

            try:
                # Create a new request with fallback model
                fallback_request = LLMRequest(
                    system_prompt=request.system_prompt,
                    user_message=request.user_message,
                    context=request.context,
                    model=fallback_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    json_mode=request.json_mode,
                    json_schema=request.json_schema,
                    metadata=request.metadata,
                )

                return await self._generate_with_model(
                    fallback_request, fallback_model, settings.fallback_llm_max_retries
                )
            except Exception as fallback_error:
                # Both primary and fallback failed
                logger.error(
                    f"Both primary ({primary_model}) and fallback ({fallback_model}) models failed. "
                    f"Primary error: {str(primary_error)}. "
                    f"Fallback error: {str(fallback_error)}"
                )
                raise Exception(
                    f"LLM generation failed for both primary and fallback models. "
                    f"Primary ({primary_model}): {str(primary_error)}. "
                    f"Fallback ({fallback_model}): {str(fallback_error)}"
                ) from fallback_error

    async def _generate_with_model(
        self, request: LLMRequest, model: str, max_retries: int
    ) -> LLMResponse:
        """
        Internal method: Generate response with a specific model and retry logic

        Args:
            request: LLM request
            model: Model to use
            max_retries: Maximum retry attempts for this model

        Returns:
            LLM response

        Raises:
            Exception: If all retry attempts fail
        """
        base_delay = settings.llm_retry_base_delay
        max_delay = settings.llm_retry_max_delay

        # Build messages for chat completion
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"{request.context}\n\nQuery: {request.user_message}"},
        ]

        # Build request payload
        payload = {
            "model": model,  # Use the specific model passed to this function
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Enable structured output if requested
        if request.json_schema:
            # Auto-format schema based on model provider for compatibility
            schema_name = request.metadata.get("schema_name", "response")
            payload["response_format"] = format_schema_for_openrouter(
                model=model,
                schema_name=schema_name,
                base_schema=request.json_schema,
            )
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

            # Enable strict parameter checking for structured outputs
            if request.json_schema:
                import json

                headers["X-OpenRouter-Provider-Preferences"] = json.dumps(
                    {"require_parameters": True}
                )

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(
            max_retries + 1
        ):  # +1 because we want max_retries RETRIES after initial attempt
            try:
                # Make request
                response = await self.client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )

                # Check for rate limiting (429)
                if response.status_code == 429:
                    error_body = response.text
                    if attempt < max_retries:
                        # Calculate exponential backoff with jitter
                        delay = min(base_delay * (2**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)  # Add 0-10% jitter
                        total_delay = delay + jitter

                        logger.warning(
                            f"Rate limit (429) on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {total_delay:.2f}s... Error: {error_body}"
                        )
                        await asyncio.sleep(total_delay)
                        continue
                    else:
                        raise Exception(
                            f"Rate limit (429) - Max retries ({max_retries}) exceeded. Response: {error_body}"
                        )

                # Check for server errors (5xx) - these are also retryable
                if 500 <= response.status_code < 600:
                    error_body = response.text
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter

                        logger.warning(
                            f"Server error ({response.status_code}) on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {total_delay:.2f}s... Error: {error_body}"
                        )
                        await asyncio.sleep(total_delay)
                        continue
                    else:
                        raise Exception(
                            f"Server error ({response.status_code}) - Max retries ({max_retries}) exceeded. "
                            f"Response: {error_body}"
                        )

                # Handle 400 errors
                if response.status_code == 400:
                    error_body = response.text
                    # Check if it's a JSON schema issue
                    if request.json_schema and (
                        "schema" in error_body.lower() or "response_format" in error_body.lower()
                    ):
                        # Schema validation errors from provider side might be transient
                        # Log warning and continue retrying
                        logger.warning(
                            f"Schema validation error from provider (attempt {attempt + 1}/{max_retries + 1}): {error_body}"
                        )
                        last_exception = Exception(
                            f"Schema validation error - Model: {model} - Response: {error_body}"
                        )
                        await asyncio.sleep(base_delay * (2**attempt))
                        continue
                    # Other 400 errors are not retryable
                    raise Exception(f"Bad request (400) - Model: {model} - Response: {error_body}")

                # Raise for other HTTP errors (not retryable)
                response.raise_for_status()

                # Parse successful response
                data = response.json()

                # Extract content and metadata
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")

                # Get token usage if available
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                # Log success if this was a retry
                if attempt > 0:
                    logger.info(f"LLM request succeeded on attempt {attempt + 1}/{max_retries + 1}")

                return LLMResponse(
                    content=content,
                    model=data.get("model", model),
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    metadata={
                        "usage": usage,
                        "response_id": data.get("id"),
                        "created": data.get("created"),
                        "retry_attempt": attempt,
                        "was_fallback": model == settings.fallback_llm_model,
                    },
                )

            except httpx.RequestError as e:
                # Network errors - retryable
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        f"Network error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    await asyncio.sleep(total_delay)
                    continue
                else:
                    raise Exception(
                        f"Network error - Max retries ({max_retries}) exceeded: {e}"
                    ) from e

            except Exception as e:
                # For other exceptions, check if they should be retried
                error_str = str(e).lower()
                is_retryable = (
                    "timeout" in error_str or "connection" in error_str or "temporary" in error_str
                )

                if is_retryable and attempt < max_retries:
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    await asyncio.sleep(total_delay)
                    last_exception = e
                    continue
                else:
                    # Not retryable or max retries exceeded
                    raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise Exception("Unexpected error: retry loop completed without success or exception")

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

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
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
