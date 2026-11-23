"""Tool contract and argument validation.

Tools are the only way an agent affects anything outside its own context, so
arguments are validated before execution, never after. A model that emits
``{"query": 42}`` gets a precise error message back and a chance to correct
itself on the next round - which is far better behaviour than a ``TypeError``
surfacing as a failed message.

Validation is a small JSON Schema subset implemented here rather than a
dependency. Tool schemas are ours, they are simple by design, and the subset
covers everything the model is allowed to send.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import InvalidToolArguments
from app.core.models import Citation
from app.inference.base import ToolSpec


@dataclass
class ToolContext:
    """Everything a tool may know about the run that invoked it."""

    channel_id: str
    message_id: str
    agent_id: str
    user_id: str = "user"
    #: Document scope requested by the user, if any.
    document_ids: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """What a tool returns.

    ``content`` is what the model reads. ``citations`` are surfaced to the UI
    and never enter the model's context as structured data - the model sees the
    numbered markers inside ``content`` instead.
    """

    content: str
    ok: bool = True
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Short line shown in the UI's tool status strip.
    summary: str = ""

    @classmethod
    def failure(cls, message: str, **metadata: Any) -> ToolResult:
        return cls(content=message, ok=False, summary=message[:120], metadata=metadata)


# --- JSON Schema subset ----------------------------------------------------

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any], *, tool_name: str
) -> dict[str, Any]:
    """Validate and coerce ``arguments``. Returns the cleaned mapping.

    Raises ``InvalidToolArguments`` with a message written for the model to
    act on.
    """
    if not isinstance(arguments, dict):
        raise InvalidToolArguments(
            f"{tool_name}: arguments must be a JSON object, got {_type_name(arguments)}"
        )

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    for name in required:
        if name not in arguments or arguments[name] is None:
            errors.append(f"missing required argument '{name}'")

    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected argument '{name}'")
            continue
        if value is None:
            continue

        error, coerced = _validate_value(value, spec, name)
        if error:
            errors.append(error)
        else:
            cleaned[name] = coerced

    for name, spec in properties.items():
        if name not in cleaned and "default" in spec:
            cleaned[name] = spec["default"]

    if errors:
        raise InvalidToolArguments(
            f"{tool_name}: " + "; ".join(errors),
            details={"tool": tool_name, "errors": errors},
        )
    return cleaned


def _validate_value(
    value: Any, spec: dict[str, Any], name: str, *, coerce: bool = True
) -> tuple[str | None, Any]:
    """Validate one value against a JSON-Schema fragment.

    ``coerce`` is lenient about the *shape* a model produced - "5" for an
    integer, a bare scalar where a list was wanted - because those are common
    and unambiguous slips. It is switched off for array items: rewriting
    ``[1]`` into ``["1"]`` silently changes the caller's data rather than
    telling the model it got the type wrong.
    """
    expected = spec.get("type")
    if expected:
        allowed = _TYPE_MAP.get(expected, ())
        # bool is a subclass of int in Python; the schema types must not blur.
        if expected in ("integer", "number") and isinstance(value, bool):
            return f"'{name}' must be a {expected}, got boolean", None
        if not isinstance(value, allowed):
            coerced = _try_coerce(value, expected) if coerce else None
            if coerced is None:
                return (
                    f"'{name}' must be a {expected}, got {_type_name(value)}",
                    None,
                )
            value = coerced

    if "enum" in spec and value not in spec["enum"]:
        return (
            f"'{name}' must be one of {spec['enum']}, got {value!r}",
            None,
        )

    if expected == "string":
        if "minLength" in spec and len(value) < spec["minLength"]:
            return f"'{name}' must be at least {spec['minLength']} characters", None
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            value = value[: spec["maxLength"]]

    if expected in ("number", "integer"):
        if "minimum" in spec and value < spec["minimum"]:
            return f"'{name}' must be >= {spec['minimum']}", None
        if "maximum" in spec and value > spec["maximum"]:
            value = spec["maximum"]

    if expected == "array":
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            value = value[: spec["maxItems"]]
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            cleaned_items = []
            for index, item in enumerate(value):
                error, coerced = _validate_value(
                    item, item_spec, f"{name}[{index}]", coerce=False
                )
                if error:
                    return error, None
                cleaned_items.append(coerced)
            value = cleaned_items

    return None, value


def _try_coerce(value: Any, expected: str) -> Any:
    try:
        if expected == "string" and isinstance(value, (int, float)):
            return str(value)
        if expected == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        if expected == "number" and isinstance(value, str):
            return float(value.strip())
        if expected == "boolean" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                return lowered == "true"
        if expected == "array" and isinstance(value, str):
            # A single value where a list was expected is a common model slip.
            return [value]
    except (TypeError, ValueError):
        return None
    return None


# --- Tool ------------------------------------------------------------------


class Tool(ABC):
    """One capability an agent can invoke."""

    #: Must match ``^[a-zA-Z0-9_-]{1,64}$`` - the providers enforce this.
    name: str = ""
    #: Written for the model, not for humans. Say when to use it and when not to.
    description: str = ""
    #: JSON Schema for the arguments object.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    #: When True, the runtime stops after this tool and streams its result
    #: instead of feeding it back to the model. Nothing uses it yet; it exists
    #: so a future "final answer" tool does not require a runtime change.
    terminal: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, parameters=self.parameters
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return validate_arguments(self.parameters, arguments, tool_name=self.name)

    @abstractmethod
    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Run the tool. Arguments have already been validated."""

    async def health(self) -> tuple[str, str | None]:
        return "healthy", None
