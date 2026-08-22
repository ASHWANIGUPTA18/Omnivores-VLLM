"""Tool validation, permissions and the guarded executor.

The executor's contract: it never raises. Whatever a tool does - throw, hang,
be missing, be forbidden - the caller gets a ``ToolResult`` the model can read
and recover from.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import InvalidToolArguments
from app.tools.base import Tool, ToolContext, ToolResult, validate_arguments
from app.tools.registry import ToolExecutor, ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2, "maxLength": 10},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "flag": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "limit": {"type": "integer", "default": 3},
    },
    "required": ["query"],
    "additionalProperties": False,
}


@pytest.fixture
def context() -> ToolContext:
    return ToolContext(channel_id="general", message_id="m1", agent_id="researcher")


# --- validation ------------------------------------------------------------


def test_valid_arguments_pass_through():
    result = validate_arguments(SCHEMA, {"query": "hello"}, tool_name="t")
    assert result["query"] == "hello"


def test_missing_required_argument_is_rejected():
    with pytest.raises(InvalidToolArguments, match="missing required argument 'query'"):
        validate_arguments(SCHEMA, {}, tool_name="t")


def test_wrong_type_is_rejected():
    with pytest.raises(InvalidToolArguments, match="must be a string"):
        validate_arguments(SCHEMA, {"query": {"a": 1}}, tool_name="t")


def test_numeric_string_is_coerced_to_integer():
    """Models emit "3" for integer fields constantly; rejecting that would
    waste a whole round on a non-problem."""
    result = validate_arguments(SCHEMA, {"query": "hi", "count": "3"}, tool_name="t")
    assert result["count"] == 3


def test_boolean_is_not_accepted_as_an_integer():
    with pytest.raises(InvalidToolArguments, match="got boolean"):
        validate_arguments(SCHEMA, {"query": "hi", "count": True}, tool_name="t")


def test_enum_violation_is_rejected():
    with pytest.raises(InvalidToolArguments, match="must be one of"):
        validate_arguments(SCHEMA, {"query": "hi", "mode": "medium"}, tool_name="t")


def test_value_below_minimum_is_rejected():
    with pytest.raises(InvalidToolArguments, match=">= 1"):
        validate_arguments(SCHEMA, {"query": "hi", "count": 0}, tool_name="t")


def test_value_above_maximum_is_clamped():
    result = validate_arguments(SCHEMA, {"query": "hi", "count": 99}, tool_name="t")
    assert result["count"] == 5


def test_overlong_string_is_truncated():
    result = validate_arguments(SCHEMA, {"query": "x" * 50}, tool_name="t")
    assert len(result["query"]) == 10


def test_unexpected_argument_is_rejected_when_disallowed():
    with pytest.raises(InvalidToolArguments, match="unexpected argument 'extra'"):
        validate_arguments(SCHEMA, {"query": "hi", "extra": 1}, tool_name="t")


def test_defaults_are_applied():
    result = validate_arguments(SCHEMA, {"query": "hi"}, tool_name="t")
    assert result["limit"] == 3


def test_array_items_are_validated():
    with pytest.raises(InvalidToolArguments):
        validate_arguments(SCHEMA, {"query": "hi", "tags": [1]}, tool_name="t")


def test_array_is_truncated_to_max_items():
    result = validate_arguments(SCHEMA, {"query": "hi", "tags": ["a", "b", "c"]}, tool_name="t")
    assert result["tags"] == ["a", "b"]


def test_scalar_is_wrapped_into_an_array():
    result = validate_arguments(SCHEMA, {"query": "hi", "tags": "a"}, tool_name="t")
    assert result["tags"] == ["a"]


def test_non_object_arguments_are_rejected():
    with pytest.raises(InvalidToolArguments, match="must be a JSON object"):
        validate_arguments(SCHEMA, ["a"], tool_name="t")


def test_all_errors_are_reported_together():
    with pytest.raises(InvalidToolArguments) as info:
        validate_arguments(SCHEMA, {"count": 0, "mode": "x"}, tool_name="t")
    assert len(info.value.details["errors"]) >= 2


# --- test doubles ----------------------------------------------------------


class EchoTool(Tool):
    name = "echo"
    description = "echoes its input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def execute(self, arguments, context):
        return ToolResult(content=arguments["text"], summary="echoed")


class ExplodingTool(Tool):
    name = "explode"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments, context):
        raise RuntimeError("boom")


class SlowTool(Tool):
    name = "slow"
    description = "never finishes in time"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments, context):
        await asyncio.sleep(10)
        return ToolResult(content="never")


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([EchoTool(), ExplodingTool(), SlowTool()])


# --- registry --------------------------------------------------------------


def test_registry_lists_names(registry):
    assert registry.names == ["echo", "explode", "slow"]


def test_specs_skip_unregistered_tools(registry):
    """An agent listing a tool this deployment lacks loses the capability,
    rather than failing to load."""
    specs = registry.specs_for(["echo", "does_not_exist"])
    assert [s.name for s in specs] == ["echo"]


def test_tool_spec_matches_the_provider_schema(registry):
    payload = registry.get("echo").spec().to_api()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "echo"
    assert payload["function"]["parameters"]["required"] == ["text"]


# --- executor --------------------------------------------------------------


async def test_executor_runs_a_tool(registry, context):
    result, metrics = await ToolExecutor(registry).execute(
        "echo", {"text": "hi"}, context
    )
    assert result.ok and result.content == "hi"
    assert metrics.ok and metrics.name == "echo"


async def test_executor_contains_an_exception(registry, context):
    result, metrics = await ToolExecutor(registry).execute("explode", {}, context)
    assert not result.ok
    assert "boom" in result.content
    assert not metrics.ok


async def test_executor_enforces_a_timeout(registry, context):
    result, metrics = await ToolExecutor(registry, timeout=0.05).execute(
        "slow", {}, context
    )
    assert not result.ok
    assert "timed out" in result.content
    assert metrics.error == "timeout"


async def test_executor_reports_an_unknown_tool(registry, context):
    result, _ = await ToolExecutor(registry).execute("nope", {}, context)
    assert not result.ok
    assert "No tool named" in result.content
    # The model is told what it *can* call, so it can correct itself.
    assert "echo" in result.content


async def test_executor_reports_invalid_arguments(registry, context):
    result, metrics = await ToolExecutor(registry).execute("echo", {}, context)
    assert not result.ok
    assert "Invalid arguments" in result.content
    assert metrics.error == "invalid_arguments"


async def test_executor_denies_a_tool_outside_the_agent_permissions(registry, context):
    result, metrics = await ToolExecutor(registry).execute(
        "explode", {}, context, allowed=["echo"]
    )
    assert not result.ok
    assert "not available to this agent" in result.content
    assert metrics.error == "denied"


async def test_executor_allows_a_permitted_tool(registry, context):
    result, _ = await ToolExecutor(registry).execute(
        "echo", {"text": "ok"}, context, allowed=["echo"]
    )
    assert result.ok
