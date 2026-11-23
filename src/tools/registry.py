"""Registry for automatic LLM tool discovery via the ``@register_tool`` decorator.

Separated from ``__init__.py`` to prevent circular imports between individual tool
modules and the package-level registry exports.
"""

from __future__ import annotations

import inspect
import logging
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}
TOOL_SCHEMAS: list[dict[str, Any]] = []

_PYTHON_TO_JSON_TYPE: dict[type[Any], str] = {
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
}


def _resolve_json_type(annotation: Any) -> str:
    """Map a Python type annotation (including Optional/Union) to a JSON Schema type."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        annotation = non_none[0] if non_none else str
    return _PYTHON_TO_JSON_TYPE.get(annotation, "string")


def _extract_parameters(func: Callable[..., Any]) -> tuple[dict[str, Any], list[str]]:
    """Build JSON Schema parameter properties and required list from function signature."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue
        json_type = _resolve_json_type(hints.get(param_name, str))
        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name} untuk {func.__name__}",
        }
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return properties, required


def register_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a Python function in ``TOOL_REGISTRY`` and append its JSON Schema.

    Args:
        func: Tool callable exposing typed parameters and a descriptive docstring.

    Returns:
        The original function unchanged.
    """
    TOOL_REGISTRY[func.__name__] = func
    properties, required = _extract_parameters(func)

    docstring = (func.__doc__ or "").strip()
    description = docstring.splitlines()[0] if docstring else func.__name__

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    TOOL_SCHEMAS.append(schema)
    logger.debug("Registered LLM audio tool: %s", func.__name__)
    return func
