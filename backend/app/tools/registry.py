"""Tool registry and executor.

The registry answers "what tools exist" and "what may this agent use". The
executor is the guarded path every tool call goes through: permission check,
argument validation, timeout, error containment, timing.

Nothing in the runtime calls ``tool.execute`` directly. Everything goes through
``ToolExecutor.execute``, which is what guarantees that a tool raising an
unexpected exception becomes a tool message the model can recover from rather
than a failed conversation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable

from app.core.config import settings
from app.core.exceptions import (
    InvalidToolArguments,
    ToolPermissionDenied,
    UnknownTool,
)
from app.core.logging import get_logger
from app.core.metrics import ToolMetrics
from app.inference.base import ToolSpec
from app.tools.base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools:
            logger.warning("replacing already-registered tool '%s'", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownTool(
                f"no tool named '{name}'",
                details={"available": sorted(self._tools)},
            ) from None

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs_for(self, tool_names: Iterable[str]) -> list[ToolSpec]:
        """Specs for the named tools, skipping any that are not registered.

        Skipping rather than raising is deliberate: an agent definition that
        lists a tool this deployment does not have should lose that capability,
        not fail to load.
        """
        specs: list[ToolSpec] = []
        for name in tool_names:
            tool = self._tools.get(name)
            if tool is None:
                logger.warning("agent requested unregistered tool '%s'", name)
                continue
            specs.append(tool.spec())
        return specs

    async def health(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for name, tool in self._tools.items():
            try:
                status, detail = await tool.health()
            except Exception as exc:  # noqa: BLE001 - health must never raise
                status, detail = "unavailable", f"{type(exc).__name__}: {exc}"
            report[name] = {"status": status, "detail": detail}
        return report


class ToolExecutor:
    """Runs one tool call safely."""

    def __init__(
        self, registry: ToolRegistry, *, timeout: float | None = None
    ) -> None:
        self.registry = registry
        self.timeout = timeout or settings.tool_timeout_seconds

    async def execute(
        self,
        name: str,
        raw_arguments: dict[str, Any],
        context: ToolContext,
        *,
        allowed: Iterable[str] | None = None,
    ) -> tuple[ToolResult, ToolMetrics]:
        started = time.perf_counter()

        def _elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        # --- permission ----------------------------------------------------
        if allowed is not None and name not in set(allowed):
            message = (
                f"Tool '{name}' is not available to this agent. "
                f"Available: {', '.join(sorted(allowed)) or 'none'}."
            )
            logger.warning(
                "blocked tool call outside agent permissions",
                extra={"tool": name, "agent_id": context.agent_id},
            )
            return (
                ToolResult.failure(message, reason=ToolPermissionDenied.code),
                ToolMetrics(name=name, duration_ms=_elapsed_ms(), ok=False, error="denied"),
            )

        # --- existence -----------------------------------------------------
        try:
            tool = self.registry.get(name)
        except UnknownTool as exc:
            available = ", ".join(exc.details.get("available", []))
            return (
                ToolResult.failure(
                    f"No tool named '{name}'. Available tools: {available}.",
                    reason=exc.code,
                ),
                ToolMetrics(name=name, duration_ms=_elapsed_ms(), ok=False, error="unknown"),
            )

        # --- arguments -----------------------------------------------------
        try:
            arguments = tool.validate(raw_arguments)
        except InvalidToolArguments as exc:
            # The model gets the exact complaint and can retry next round.
            return (
                ToolResult.failure(
                    f"Invalid arguments. {exc.message}. Fix them and call the tool "
                    "again.",
                    reason=exc.code,
                ),
                ToolMetrics(
                    name=name, duration_ms=_elapsed_ms(), ok=False, error="invalid_arguments"
                ),
            )

        # --- run -----------------------------------------------------------
        try:
            result = await asyncio.wait_for(
                tool.execute(arguments, context), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning("tool '%s' timed out after %.0fs", name, self.timeout)
            return (
                ToolResult.failure(
                    f"Tool '{name}' timed out after {self.timeout:.0f}s. Continue "
                    "without its result.",
                    reason="tool_timeout",
                ),
                ToolMetrics(
                    name=name, duration_ms=_elapsed_ms(), ok=False, error="timeout"
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - containment is the point
            logger.exception("tool '%s' raised", name)
            return (
                ToolResult.failure(
                    f"Tool '{name}' failed: {type(exc).__name__}: {exc}",
                    reason="tool_execution_error",
                ),
                ToolMetrics(
                    name=name,
                    duration_ms=_elapsed_ms(),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

        duration_ms = _elapsed_ms()
        logger.debug(
            "tool '%s' finished in %.0fms (ok=%s)", name, duration_ms, result.ok
        )
        return result, ToolMetrics(name=name, duration_ms=duration_ms, ok=result.ok)


def build_default_registry() -> ToolRegistry:
    """The tools every deployment gets.

    Constructed eagerly but with lazy internals - creating ``PythonTool`` does
    not create a sandbox, and creating ``KnowledgeBaseTool`` does not load an
    embedding model.
    """
    from app.tools.knowledge_base import KnowledgeBaseTool
    from app.tools.python_exec import PythonTool
    from app.tools.web_search import WebSearchTool

    return ToolRegistry([KnowledgeBaseTool(), PythonTool(), WebSearchTool()])


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = build_default_registry()
        logger.info("tool registry ready", extra={"tools": _registry.names})
    return _registry


def set_tool_registry(registry: ToolRegistry | None) -> None:
    global _registry
    _registry = registry
