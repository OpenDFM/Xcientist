"""
JSON extraction and conversion utilities for agent outputs.

This module provides functions to extract JSON from model outputs and convert them
to Pydantic schema classes, replacing the previous unifier agent pattern.
"""

import json
import re
from typing import Type, TypeVar, Optional, Any, Tuple
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class JSONParseError(Exception):
    """Custom exception for JSON parsing failures that should trigger retry."""
    
    def __init__(self, message: str, raw_text: str = "", recoverable: bool = True):
        super().__init__(message)
        self.raw_text = raw_text
        self.recoverable = recoverable  # If True, the step should be retried


def clean_json_string(json_str: str) -> str:
    """
    Clean and fix common JSON issues from LLM outputs.
    
    Handles:
    - Trailing commas before } or ]
    - JavaScript-style comments (// and /* */)
    - Unescaped control characters
    - BOM and invisible characters
    - Smart quotes replacement
    - Incomplete escape sequences
    
    Args:
        json_str: Raw JSON string that may have issues
        
    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str
    
    # Remove BOM and invisible characters
    json_str = json_str.strip("\ufeff\u200b\u200c\u200d\u2060")
    
    # Replace smart quotes with standard quotes
    json_str = json_str.replace(""", "\"").replace(""", "\"")
    json_str = json_str.replace("'", "'").replace("'", "'")
    
    # Remove JavaScript-style single-line comments (// ...)
    # Be careful not to match URLs (http://)
    json_str = re.sub(r"(?<!:)//[^\n]*", "", json_str)
    
    # Remove JavaScript-style multi-line comments (/* ... */)
    json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    
    # Fix unescaped newlines inside strings (common LLM issue)
    # This is tricky - we need to find strings and escape newlines inside them
    # Simple approach: replace literal newlines that aren't already escaped
    def escape_string_newlines(match):
        content = match.group(1)
        # Escape unescaped newlines
        content = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return '"' + content + '"'
    
    # Try to fix strings with unescaped newlines
    # This pattern matches strings but may not be perfect for all cases
    try:
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)(?:\n|\r\n?)((?:[^"\\]|\\.)*)"', 
                         lambda m: '"' + m.group(1).replace("\n", "\\n") + "\\n" + m.group(2).replace("\n", "\\n") + '"', 
                         json_str)
    except Exception:
        pass  # If regex fails, continue with original
    
    return json_str


def find_json_boundaries(text: str) -> list[Tuple[int, int]]:
    """
    Find all potential JSON object boundaries in text.
    
    Returns list of (start, end) tuples for each potential JSON object,
    sorted by length (longest first) to prefer complete JSON.
    
    Args:
        text: Text to search for JSON
        
    Returns:
        List of (start_index, end_index) tuples
    """
    boundaries = []
    stack = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        # Skip characters inside strings
        if char == '"':
            i += 1
            while i < len(text):
                if text[i] == '"' and text[i-1] != '\\':
                    break
                i += 1
        elif char == '{':
            stack.append(i)
        elif char == '}' and stack:
            start = stack.pop()
            boundaries.append((start, i + 1))
        
        i += 1
    
    # Sort by length (longest first) to prefer complete JSON
    boundaries.sort(key=lambda x: x[1] - x[0], reverse=True)
    return boundaries


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON content from model output text.
    
    Handles various formats:
    - JSON wrapped in ```json ... ``` code blocks
    - JSON wrapped in ``` ... ``` code blocks
    - Raw JSON object starting with { and ending with }
    - JSON embedded in other text
    - Multiple JSON attempts (tries cleaning and repairs)
    
    Args:
        text: Raw text from model output
        
    Returns:
        Extracted JSON string, or None if no valid JSON found
    """
    if not text:
        return None
    
    # Strategy 1: Try to find JSON in code blocks first (```json ... ``` or ``` ... ```)
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, text, re.IGNORECASE)
    
    for match in matches:
        match = match.strip()
        if match.startswith("{") and "}" in match:
            # Try raw first
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                pass
            
            # Try with cleaning
            cleaned = clean_json_string(match)
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Find all potential JSON boundaries and try each
    boundaries = find_json_boundaries(text)
    
    for start, end in boundaries:
        potential_json = text[start:end]
        
        # Try raw first
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass
        
        # Try with cleaning
        cleaned = clean_json_string(potential_json)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Try to find raw JSON object using brace matching (fallback)
    first_brace = text.find("{")
    if first_brace != -1:
        # Find matching closing brace
        brace_count = 0
        last_brace = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[first_brace:], start=first_brace):
            if escape_next:
                escape_next = False
                continue
            
            if char == "\\" and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace = i
                        break
        
        if last_brace != -1:
            potential_json = text[first_brace:last_brace + 1]
            
            # Try raw
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
            
            # Try cleaned
            cleaned = clean_json_string(potential_json)
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                pass
    
    # Strategy 4: Try to repair truncated JSON
    repaired = try_repair_truncated_json(text)
    if repaired:
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    
    return None


def try_repair_truncated_json(text: str) -> Optional[str]:
    """
    Attempt to repair truncated or incomplete JSON.
    
    Common cases:
    - Missing closing braces/brackets
    - Truncated string values
    
    Args:
        text: Text containing potentially truncated JSON
        
    Returns:
        Repaired JSON string or None if repair not possible
    """
    # Find JSON start
    first_brace = text.find("{")
    if first_brace == -1:
        return None
    
    json_text = text[first_brace:]
    
    # Count unclosed braces and brackets
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0
    
    for i, char in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\" and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count >= 0:
                    last_valid_pos = i
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count >= 0:
                    last_valid_pos = i
    
    if brace_count == 0 and bracket_count == 0:
        return None  # JSON is complete, no repair needed
    
    # Try to repair by adding missing closures
    repaired = json_text[:last_valid_pos + 1] if last_valid_pos > 0 else json_text
    
    # If we're in a string, close it
    if in_string:
        repaired += '"'
    
    # Add missing brackets and braces
    repaired += "]" * max(0, bracket_count)
    repaired += "}" * max(0, brace_count)
    
    return repaired if repaired != json_text else None


def parse_json_to_schema(json_str: str, schema_class: Type[T]) -> T:
    """
    Parse JSON string and convert to a Pydantic schema class instance.
    
    Args:
        json_str: Valid JSON string
        schema_class: Pydantic model class to convert to
        
    Returns:
        Instance of the schema class
        
    Raises:
        json.JSONDecodeError: If JSON is invalid
        pydantic.ValidationError: If JSON doesn't match schema
    """
    data = json.loads(json_str)
    return schema_class(**data)


def extract_and_parse_json(
    text: str, 
    schema_class: Type[T],
    default_factory: Optional[callable] = None,
    raise_on_failure: bool = False
) -> T:
    """
    Extract JSON from text and parse to schema class.
    
    This is the main function to use for replacing unifier agents.
    
    Args:
        text: Raw text from model output
        schema_class: Pydantic model class to convert to
        default_factory: Optional function to create default instance if extraction fails
        raise_on_failure: If True, raise JSONParseError on failure (for retry logic)
        
    Returns:
        Instance of the schema class
        
    Raises:
        JSONParseError: If raise_on_failure=True and parsing fails (should trigger retry)
        ValueError: If no valid JSON found and no default_factory provided
        pydantic.ValidationError: If JSON doesn't match schema
    """
    json_str = extract_json_from_text(text)
    
    if json_str is None:
        error_msg = f"No valid JSON found in text. Text preview: {text[:500] if text else 'empty'}..."
        if raise_on_failure:
            raise JSONParseError(error_msg, raw_text=text, recoverable=True)
        if default_factory:
            return default_factory()
        raise ValueError(error_msg)
    
    try:
        return parse_json_to_schema(json_str, schema_class)
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}. JSON preview: {json_str[:300]}..."
        if raise_on_failure:
            raise JSONParseError(error_msg, raw_text=text, recoverable=True)
        raise
    except Exception as e:
        error_msg = f"Schema validation error: {str(e)}"
        if raise_on_failure:
            raise JSONParseError(error_msg, raw_text=text, recoverable=True)
        raise


def generate_json_schema_instruction(schema_class: Type[BaseModel]) -> str:
    """
    Generate instruction text describing the expected JSON output format
    based on a Pydantic schema class.
    
    Args:
        schema_class: Pydantic model class
        
    Returns:
        Formatted instruction string describing the JSON schema
    """
    schema = schema_class.model_json_schema()
    
    def format_property(name: str, prop: dict, required: list) -> str:
        """Format a single property description."""
        prop_type = prop.get('type', 'any')
        description = prop.get('description', 'No description')
        is_required = name in required
        default = prop.get('default', None)
        
        # Handle complex types
        if prop_type == 'array':
            items = prop.get('items', {})
            item_type = items.get('type', 'any')
            if '$ref' in items:
                ref_name = items['$ref'].split('/')[-1]
                item_type = ref_name
            prop_type = f"List[{item_type}]"
        elif '$ref' in prop:
            ref_name = prop['$ref'].split('/')[-1]
            prop_type = ref_name
        
        required_str = "REQUIRED" if is_required else f"optional, default={default}"
        return f"  - {name} ({prop_type}, {required_str}): {description}"
    
    # Build instruction
    lines = [
        f"## Output Format: {schema_class.__name__}",
        "",
        "You MUST output your response as a JSON object wrapped in ```json ... ``` code blocks.",
        "The JSON MUST conform to the following schema:",
        "",
        "### Fields:",
    ]
    
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    for name, prop in properties.items():
        lines.append(format_property(name, prop, required))
    
    # Add nested schemas if any
    definitions = schema.get('$defs', {})
    if definitions:
        lines.append("")
        lines.append("### Nested Types:")
        for def_name, def_schema in definitions.items():
            lines.append(f"\n#### {def_name}:")
            def_props = def_schema.get('properties', {})
            def_required = def_schema.get('required', [])
            for name, prop in def_props.items():
                lines.append(format_property(name, prop, def_required))
    
    # Generate a concrete example based on required fields
    example_lines = ["{"]
    for name in required:
        prop = properties.get(name, {})
        prop_type = prop.get('type', 'string')
        if prop_type == 'string':
            example_lines.append(f'  "{name}": "...",')
        elif prop_type == 'boolean':
            example_lines.append(f'  "{name}": false,')
        elif prop_type == 'integer' or prop_type == 'number':
            example_lines.append(f'  "{name}": 0,')
        elif prop_type == 'array':
            example_lines.append(f'  "{name}": [],')
        elif prop_type == 'object' or '$ref' in prop:
            example_lines.append(f'  "{name}": {{}},')
        else:
            example_lines.append(f'  "{name}": null,')
    # Remove trailing comma from last line
    if example_lines[-1].endswith(','):
        example_lines[-1] = example_lines[-1][:-1]
    example_lines.append("}")
    example_str = "\n".join(example_lines)
    
    lines.extend([
        "",
        "### Example Output Format:",
        "```json",
        example_str,
        "```",
        "",
        "IMPORTANT: Your JSON output MUST include ALL required fields listed above.",
        "Return the FULL top-level JSON structure.",
    ])
    
    return "\n".join(lines)


def safe_extract_and_parse(
    text: str,
    schema_class: Type[T],
    error_handler: Optional[callable] = None
) -> Tuple[Optional[T], Optional[str], bool]:
    """
    Safely extract and parse JSON with error handling.
    
    Args:
        text: Raw text from model output
        schema_class: Pydantic model class to convert to
        error_handler: Optional callback for error logging
        
    Returns:
        Tuple of (parsed_result or None, error_message or None, should_retry: bool)
        should_retry is True if the parsing failed and the step should be retried
    """
    try:
        result = extract_and_parse_json(text, schema_class)
        return result, None, False
    except JSONParseError as e:
        error_msg = f"JSON parse error (recoverable): {str(e)}"
        if error_handler:
            error_handler(error_msg)
        return None, error_msg, e.recoverable
    except ValueError as e:
        error_msg = f"JSON extraction failed: {str(e)}"
        if error_handler:
            error_handler(error_msg)
        return None, error_msg, True  # Should retry
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing failed: {str(e)}"
        if error_handler:
            error_handler(error_msg)
        return None, error_msg, True  # Should retry
    except Exception as e:
        error_msg = f"Schema validation failed: {str(e)}"
        if error_handler:
            error_handler(error_msg)
        return None, error_msg, True  # Should retry


def extract_and_parse_json_with_retry_info(
    text: str, 
    schema_class: Type[T],
    default_factory: Optional[callable] = None
) -> Tuple[Optional[T], bool, Optional[str]]:
    """
    Extract and parse JSON, returning retry information.
    
    This function is designed for the experiment workflow to determine
    if a step should be retried due to JSON parsing failure.
    
    Args:
        text: Raw text from model output
        schema_class: Pydantic model class to convert to
        default_factory: Optional function to create default instance if extraction fails
        
    Returns:
        Tuple of (result, should_retry, error_message)
        - result: Parsed schema instance, or default instance if default_factory provided
        - should_retry: True if parsing failed and step should be retried
        - error_message: Error description if parsing failed, None otherwise
    """
    json_str = extract_json_from_text(text)
    
    if json_str is None:
        error_msg = f"No valid JSON found in text. Text length: {len(text) if text else 0}"
        if default_factory:
            return default_factory(), True, error_msg
        return None, True, error_msg
    
    try:
        result = parse_json_to_schema(json_str, schema_class)
        return result, False, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error at position {e.pos}: {e.msg}"
        if default_factory:
            return default_factory(), True, error_msg
        return None, True, error_msg
    except Exception as e:
        error_msg = f"Schema validation error: {str(e)}"
        if default_factory:
            return default_factory(), True, error_msg
        return None, True, error_msg

