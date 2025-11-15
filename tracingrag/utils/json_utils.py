"""Utility functions for cleaning and parsing JSON from LLM responses"""

import json
import re
from typing import Any

from tracingrag.utils.logger import get_logger

logger = get_logger(__name__)


def clean_llm_json(content: str) -> str:
    """Clean JSON response from LLM by removing markdown code blocks and extra text

    Handles common LLM output issues:
    - Markdown code blocks: ```json ... ``` or ``` ... ```
    - Leading/trailing text
    - Extra whitespace

    Args:
        content: Raw LLM response content

    Returns:
        Cleaned JSON string ready for parsing

    Examples:
        >>> clean_llm_json('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> clean_llm_json('Here is the result:\\n{"key": "value"}')
        '{"key": "value"}'
    """
    if not content:
        return content

    # Strip whitespace
    cleaned = content.strip()

    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    # Pattern matches: optional "```json" or "```", content, optional "```"
    code_block_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```\s*$"
    match = re.match(code_block_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
        logger.debug("Removed markdown code block from JSON response")

    # If still has code block markers, try removing them directly
    if cleaned.startswith("```"):
        # Find first newline after opening ```
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        else:
            # No newline, just remove ```json or ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)

    # Remove trailing code block markers (handle ``` or **```** or ```** etc)
    cleaned = re.sub(r"\*?\*?```\*?\*?\s*$", "", cleaned).rstrip()

    # Remove leading text before JSON (find first { or [)
    if "{" in cleaned or "[" in cleaned:
        json_start = min(
            cleaned.find("{") if "{" in cleaned else len(cleaned),
            cleaned.find("[") if "[" in cleaned else len(cleaned),
        )
        if json_start > 0:
            logger.debug(f"Removed {json_start} chars of leading text before JSON")
            cleaned = cleaned[json_start:]

    # Remove trailing text after JSON (find last } or ])
    if "}" in cleaned or "]" in cleaned:
        json_end = max(
            cleaned.rfind("}") if "}" in cleaned else -1,
            cleaned.rfind("]") if "]" in cleaned else -1,
        )
        if json_end != -1 and json_end < len(cleaned) - 1:
            logger.debug(f"Removed {len(cleaned) - json_end - 1} chars of trailing text after JSON")
            cleaned = cleaned[: json_end + 1]

    # Remove trailing commas before ] or } (invalid in JSON)
    # Pattern: ,\s*] or ,\s*}
    original = cleaned
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    if cleaned != original:
        logger.debug("Removed trailing commas from JSON")

    return cleaned.strip()


def fix_malformed_json(content: str) -> str:
    """Fix common malformed JSON issues from LLM outputs

    Handles issues like:
    - Missing colons: "new [ -> "new": [
    - Missing quotes in object keys: {index 4 -> {"index": 4
    - Missing quotes in object values (numbers are ok)

    Args:
        content: Potentially malformed JSON string

    Returns:
        Fixed JSON string
    """
    fixed = content

    # Fix 1: Missing colon after property name
    # Pattern: "property_name" [ or "property_name" {
    # Should be: "property_name": [ or "property_name": {
    fixed = re.sub(
        r'("(?:new|existing|properties|items|[a-zA-Z_][a-zA-Z0-9_]*)")\s*([{\[])', r"\1: \2", fixed
    )

    # Fix 2: Missing quotes around property names
    # Pattern: {property_name: -> {"property_name":
    # Match word characters after { or , followed by :
    fixed = re.sub(r"([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', fixed)

    # Fix 3: Missing closing quote for string values
    # Pattern: "key": "value (EOF or ,) -> "key": "value",
    # This is tricky - we'll count quotes and add missing ones at logical boundaries

    # Fix 4: Missing comma between array elements
    # Pattern: } { -> }, {
    fixed = re.sub(r"}\s*{", r"}, {", fixed)

    return fixed


def parse_llm_json(
    content: str,
    strict: bool = False,
    fix_incomplete: bool = True,
    fix_malformed: bool = True,
) -> dict[str, Any] | list[Any] | None:
    """Parse JSON from LLM response with automatic cleaning and error recovery

    Args:
        content: Raw LLM response content
        strict: If True, use strict JSON parsing (default: False for lenient parsing)
        fix_incomplete: If True, attempt to fix incomplete JSON (missing closing brackets/braces)
        fix_malformed: If True, attempt to fix malformed JSON (missing colons, quotes, etc.)

    Returns:
        Parsed JSON object (dict or list), or None if parsing fails

    Raises:
        json.JSONDecodeError: Only if all recovery attempts fail and strict=True
    """
    if not content or not content.strip():
        logger.warning("Empty content provided to parse_llm_json")
        return None

    # First attempt: direct parsing
    try:
        return json.loads(content, strict=strict)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")

    # Second attempt: clean and parse
    try:
        cleaned = clean_llm_json(content)
        return json.loads(cleaned, strict=strict)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parsing failed after cleaning: {e}")

    # Third attempt: fix malformed JSON issues (new!)
    if fix_malformed:
        try:
            cleaned = clean_llm_json(content)
            fixed = fix_malformed_json(cleaned)
            logger.debug("Applied malformed JSON fixes")
            return json.loads(fixed, strict=strict)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed after malformed fixes: {e}")

    # Fourth attempt: fix incomplete JSON (missing closing brackets/braces)
    if fix_incomplete:
        try:
            cleaned = clean_llm_json(content)

            # Count brackets and braces
            open_braces = cleaned.count("{")
            close_braces = cleaned.count("}")
            open_brackets = cleaned.count("[")
            close_brackets = cleaned.count("]")

            if open_braces > close_braces or open_brackets > close_brackets:
                logger.debug(
                    f"Detected incomplete JSON: "
                    f"braces={open_braces}/{close_braces}, "
                    f"brackets={open_brackets}/{close_brackets}"
                )

                # Add missing closing quotes if string is unterminated
                if cleaned.count('"') % 2 != 0:
                    cleaned += '"'
                    logger.debug("Added missing closing quote")

                # Close missing brackets
                while open_brackets > close_brackets:
                    cleaned += "]"
                    close_brackets += 1
                    logger.debug("Added missing ]")

                # Close missing braces
                while open_braces > close_braces:
                    cleaned += "}"
                    close_braces += 1
                    logger.debug("Added missing }")

                return json.loads(cleaned, strict=strict)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to fix incomplete JSON: {e}")

    # Fifth attempt: combine malformed + incomplete fixes
    if fix_malformed and fix_incomplete:
        try:
            cleaned = clean_llm_json(content)
            fixed = fix_malformed_json(cleaned)

            # Count brackets and braces
            open_braces = fixed.count("{")
            close_braces = fixed.count("}")
            open_brackets = fixed.count("[")
            close_brackets = fixed.count("]")

            if open_braces > close_braces or open_brackets > close_brackets:
                logger.debug("Applying combined malformed + incomplete fixes")

                # Add missing closing quotes if string is unterminated
                if fixed.count('"') % 2 != 0:
                    fixed += '"'

                # Close missing brackets
                while open_brackets > close_brackets:
                    fixed += "]"
                    close_brackets += 1

                # Close missing braces
                while open_braces > close_braces:
                    fixed += "}"
                    close_braces += 1

            return json.loads(fixed, strict=strict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed combined malformed + incomplete fixes: {e}")

    # All attempts failed
    logger.error("All JSON parsing attempts failed")
    logger.error(f"Original content (first 500 chars): {content[:500]}")
    logger.error(f"Original content (last 200 chars): {content[-200:]}")
    return None
