"""Code execution sandboxing.

Running model-generated Python is the single most dangerous thing this
application does. The rule enforced here: **the backend process never executes
model-generated code in its own interpreter.** There is no ``exec()`` anywhere
in this package.

Three modes, selected by ``PYTHON_SANDBOX_MODE``:

``disabled`` (default)
    The tool refuses and explains how to enable it. Safe by construction, and
    the right default for a repository someone might run without reading the
    docs first.

``docker``
    Each execution gets a fresh container: no network, read-only root
    filesystem, dropped capabilities, memory/CPU/PID limits, non-root user, and
    a hard timeout. This is the production mode.

``subprocess``
    A separate interpreter process with a timeout and output caps. It is
    **not** a security boundary - the code can read your files and reach your
    network. Development convenience only; the config layer warns when it is
    enabled in production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False
    #: True when output was cut to the configured cap.
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def format_for_model(self) -> str:
        """Render the result as the tool message the model will read."""
        parts: list[str] = []
        if self.timed_out:
            parts.append(f"EXECUTION TIMED OUT after {self.duration_ms / 1000:.1f}s")
        if self.stdout.strip():
            parts.append("stdout:\n" + self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("stderr:\n" + self.stderr.rstrip())
        if not self.timed_out and self.exit_code != 0:
            parts.append(f"process exited with code {self.exit_code}")
        if self.truncated:
            parts.append("(output truncated)")
        if not parts:
            parts.append(
                "The code ran successfully and produced no output. "
                "Use print() to return a value."
            )
        return "\n\n".join(parts)


class ExecutionSandbox(ABC):
    """Runs untrusted code and returns what it printed."""

    name: str = "base"

    @abstractmethod
    async def run_python(self, code: str, *, timeout: float | None = None) -> SandboxResult:
        ...

    @abstractmethod
    async def health(self) -> tuple[str, str | None]:
        """Return ``(status, detail)`` without raising."""

    async def close(self) -> None:
        return None
