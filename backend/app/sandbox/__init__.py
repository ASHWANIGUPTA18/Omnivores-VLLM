"""Sandbox selection.

``disabled`` is the default and is implemented as a real sandbox that always
refuses, rather than as a ``None`` the callers have to check for. That keeps
the tool layer free of special cases and guarantees the refusal message reaches
the model and the user.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.exceptions import ConfigurationError, SandboxDisabled
from app.core.logging import get_logger
from app.sandbox.base import ExecutionSandbox, SandboxResult

logger = get_logger(__name__)

__all__ = [
    "ExecutionSandbox",
    "SandboxResult",
    "DisabledSandbox",
    "create_sandbox",
    "get_sandbox",
    "set_sandbox",
]


class DisabledSandbox(ExecutionSandbox):
    name = "disabled"

    async def run_python(self, code: str, *, timeout: float | None = None) -> SandboxResult:
        raise SandboxDisabled(
            "python execution is disabled (PYTHON_SANDBOX_MODE=disabled)"
        )

    async def health(self) -> tuple[str, str | None]:
        return "healthy", "code execution disabled by configuration"


def create_sandbox(mode: str | None = None) -> ExecutionSandbox:
    selected = (mode or settings.python_sandbox_mode).lower()

    if selected == "disabled":
        return DisabledSandbox()
    if selected == "docker":
        from app.sandbox.docker_sandbox import DockerSandbox

        return DockerSandbox()
    if selected == "subprocess":
        from app.sandbox.subprocess_sandbox import SubprocessSandbox

        return SubprocessSandbox()

    raise ConfigurationError(
        f"unknown PYTHON_SANDBOX_MODE '{selected}'",
        details={"supported": ["disabled", "docker", "subprocess"]},
    )


_sandbox: ExecutionSandbox | None = None


def get_sandbox() -> ExecutionSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = create_sandbox()
        logger.info("sandbox ready", extra={"mode": _sandbox.name})
    return _sandbox


def set_sandbox(sandbox: ExecutionSandbox | None) -> None:
    global _sandbox
    _sandbox = sandbox
