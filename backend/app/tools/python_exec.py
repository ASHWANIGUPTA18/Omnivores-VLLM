"""Python execution tool.

Thin by design: all the danger is handled in ``app/sandbox``. This module's
only jobs are describing the tool to the model in a way that produces runnable
code, and turning a ``SandboxResult`` into a useful tool message.

Note what the description tells the model: print your results. Without that,
models routinely write an expression on the last line and expect a REPL to echo
it, then report success having returned nothing.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.exceptions import SandboxDisabled, SandboxError, SandboxUnavailable
from app.core.logging import get_logger
from app.sandbox import ExecutionSandbox, get_sandbox
from app.tools.base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)


class PythonTool(Tool):
    name = "run_python"
    description = (
        "Execute Python 3 code in an isolated sandbox and return what it "
        "prints. Use it for calculations, data manipulation, parsing, and "
        "verifying algorithms - never guess at arithmetic you could run. "
        "The sandbox has no network access and no filesystem access outside a "
        "temporary directory, and only the standard library is guaranteed to be "
        "available. You must print() anything you want to see; the last "
        "expression is not echoed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Complete, self-contained Python 3 source. Include every "
                    "import. End with print() calls for the results."
                ),
                "minLength": 1,
                "maxLength": 20000,
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox: ExecutionSandbox | None = None) -> None:
        self._sandbox = sandbox

    @property
    def sandbox(self) -> ExecutionSandbox:
        return self._sandbox or get_sandbox()

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        code = arguments["code"]

        try:
            result = await self.sandbox.run_python(
                code, timeout=settings.python_timeout_seconds
            )
        except SandboxDisabled as exc:
            # Tell the model plainly so it answers without the tool rather than
            # retrying the same call every round.
            return ToolResult.failure(
                "Code execution is disabled on this server. Answer the question "
                "without running code, and say that you could not execute it.",
                reason=exc.code,
            )
        except SandboxUnavailable as exc:
            return ToolResult.failure(
                f"The code sandbox is unavailable ({exc.message}). Answer without "
                "running code.",
                reason=exc.code,
            )
        except SandboxError as exc:
            logger.error("sandbox error: %s", exc)
            return ToolResult.failure(f"Code execution failed: {exc.user_message}")

        content = result.format_for_model()

        if result.timed_out:
            summary = f"Timed out after {settings.python_timeout_seconds:.0f}s"
        elif result.ok:
            summary = f"Ran in {result.duration_ms:.0f}ms"
        else:
            summary = f"Exited with code {result.exit_code}"

        return ToolResult(
            content=content,
            ok=result.ok,
            summary=summary,
            metadata={
                "exit_code": result.exit_code,
                "duration_ms": round(result.duration_ms, 1),
                "timed_out": result.timed_out,
                "truncated": result.truncated,
                "sandbox": self.sandbox.name,
                # Echoed back so the UI can show what actually ran.
                "code": code,
            },
        )

    async def health(self) -> tuple[str, str | None]:
        return await self.sandbox.health()
