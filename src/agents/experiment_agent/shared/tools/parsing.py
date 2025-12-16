"""
Parsing Utilities - Unified LLM Output Parsing

Provides:
- JSON extraction from LLM outputs
- Code block extraction
- Schema validation helpers

Used by all agent types for parsing structured outputs.
"""

import json
import re
import logging
from typing import Optional, Dict, Any, TypeVar, Type

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_from_llm_output(output: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM output.

    Handles multiple formats:
    1. ```json ... ``` blocks
    2. Raw JSON objects
    3. JSON embedded in text

    Args:
        output: Raw LLM output string

    Returns:
        Parsed JSON as dictionary, or None if extraction failed
    """
    if not output:
        return None

    json_data = None

    # Method 1: Look for ```json ... ``` block
    if "```json" in output:
        try:
            json_str = output.split("```json", 1)[1]
            if "```" in json_str:
                json_str = json_str.split("```", 1)[0]
            json_data = json.loads(json_str.strip())
            return json_data
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"Failed to parse ```json block: {e}")

    # Method 2: Look for ``` ... ``` block (without json tag)
    if json_data is None and "```" in output:
        try:
            # Find content between first ``` and next ```
            match = re.search(r"```\s*\n?(.*?)\n?```", output, re.DOTALL)
            if match:
                potential_json = match.group(1).strip()
                if potential_json.startswith("{"):
                    json_data = json.loads(potential_json)
                    return json_data
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"Failed to parse ``` block: {e}")

    # Method 3: Find JSON object directly using brace matching
    if json_data is None:
        start = output.find("{")
        if start != -1:
            depth = 0
            end = start
            in_string = False
            escape_next = False

            for i, char in enumerate(output[start:], start):
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                try:
                    json_data = json.loads(output[start:end])
                    return json_data
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON object: {e}")

    return None


def extract_code_block(output: str, language: str = "python") -> Optional[str]:
    """
    Extract code block from LLM output.

    Args:
        output: Raw LLM output string
        language: Expected language tag (e.g., "python", "bash")

    Returns:
        Code content, or None if not found
    """
    if not output:
        return None

    # Try language-specific block first
    pattern = f"```{language}\\s*\\n(.*?)\\n?```"
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code block
    match = re.search(r"```\s*\n(.*?)\n?```", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def parse_to_model(
    output: str,
    model_class: Type[T],
    partial_ok: bool = False,
) -> Optional[T]:
    """
    Parse LLM output to a Pydantic model.

    Args:
        output: Raw LLM output string
        model_class: Target Pydantic model class
        partial_ok: If True, try to create model with partial data on validation error

    Returns:
        Parsed model instance, or None if parsing failed
    """
    json_data = extract_json_from_llm_output(output)

    if json_data is None:
        logger.warning(f"Could not extract JSON from output for {model_class.__name__}")
        return None

    try:
        return model_class(**json_data)
    except ValidationError as e:
        logger.warning(f"Validation error for {model_class.__name__}: {e}")

        if partial_ok:
            # Try to create with only valid fields
            valid_fields = {}
            for field_name in model_class.model_fields:
                if field_name in json_data:
                    valid_fields[field_name] = json_data[field_name]

            try:
                return model_class(**valid_fields)
            except ValidationError:
                pass

        return None


def extract_verdict(output: str) -> Optional[bool]:
    """
    Extract pass/fail verdict from LLM output.

    Args:
        output: Raw LLM output string

    Returns:
        True for pass, False for fail, None if unclear
    """
    output_lower = output.lower()

    # Check explicit verdicts
    if "verdict: pass" in output_lower or "verdict:pass" in output_lower:
        return True
    if "verdict: fail" in output_lower or "verdict:fail" in output_lower:
        return False

    # Check error counts
    if "errors: 0" in output_lower or "errors:0" in output_lower:
        return True

    # Check for failure indicators
    if any(word in output_lower for word in ["error", "failed", "failure"]):
        return False

    # Check for success indicators
    if any(word in output_lower for word in ["success", "passed", "complete"]):
        return True

    return None


def extract_status(output: str) -> Optional[str]:
    """
    Extract status from LLM output.

    Args:
        output: Raw LLM output string

    Returns:
        Status string ("success" or "failure"), or None
    """
    output_upper = output.upper()

    if "SUCCESS" in output_upper and "FAILURE" not in output_upper:
        return "success"
    if "FAILURE" in output_upper or "FAILED" in output_upper:
        return "failure"

    return None


def clean_llm_output(output: str) -> str:
    """
    Clean LLM output by removing common artifacts.

    Args:
        output: Raw LLM output string

    Returns:
        Cleaned output string
    """
    if not output:
        return ""

    # Remove thinking tags
    output = re.sub(r"<thinking>.*?</thinking>", "", output, flags=re.DOTALL)

    # Remove extra whitespace
    output = re.sub(r"\n{3,}", "\n\n", output)

    return output.strip()
