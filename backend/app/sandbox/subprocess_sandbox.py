"""Subprocess execution sandbox - DEVELOPMENT ONLY.

This is not a security boundary and must never be treated as one. Code run here
executes with the backend process's own user, and can read your files, reach
your network, and call anything the interpreter can. It exists so the coder
agent is demonstrable on a laptop that has no Docker, and it announces itself
loudly.

What it does provide, and should not be mistaken for isolation:

* a separate process, so a crash or ``sys.exit`` cannot take down the server
* ``-I`` isolated mode, so the caller's site-packages and env vars do not leak
* a wall-clock timeout with a process-tree kill
* output caps
* a temporary working directory, removed afterwards
* POSIX resource limits (address space, CPU, file size) where available

Use ``PYTHON_SANDBOX_MODE=docker`` for anything real.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time

from app.core.config import settings
from app.core.logging import get_logger
from app.sandbox.base import ExecutionSandbox, SandboxResult

logger = get_logger(__name__)

#: Applied inside the child before the user's code runs, on platforms that
#: support the resource module (not Windows).
_PREAMBLE = """\
import sys
try:
    import resource
    _MEM = {memory_bytes}
    resource.setrlimit(resource.RLIMIT_AS, (_MEM, _MEM))
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
except Exception:
    pass
sys.argv = ['sandbox']
"""


class SubprocessSandbox(ExecutionSandbox):
    name = "subprocess"

    def __init__(
        self,
        *,
        timeout: float | None = None,
        memory_limit_mb: int | None = None,
        max_output_chars: int | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.timeout = timeout or settings.python_timeout_seconds
        self.memory_limit_mb = memory_limit_mb or settings.python_memory_limit_mb
        self.max_output_chars = max_output_chars or settings.python_max_output_chars
        self.python_executable = python_executable or sys.executable
        logger.warning(
            "PYTHON_SANDBOX_MODE=subprocess - model-generated code will run on this "
            "host with the server's own privileges. This is not isolation. Use "
            "PYTHON_SANDBOX_MODE=docker outside local development."
        )

    def _wrap(self, code: str) -> str:
        preamble = _PREAMBLE.format(
            memory_bytes=self.memory_limit_mb * 1024 * 1024,
            cpu_seconds=int(max(1, self.timeout)),
        )
        return preamble + "\n" + code

    async def run_python(self, code: str, *, timeout: float | None = None) -> SandboxResult:
        limit = timeout or self.timeout
        started = time.perf_counter()
        workdir = tempfile.mkdtemp(prefix="omnivores-sandbox-")

        # A minimal environment: no inherited secrets, no PYTHONPATH.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": workdir,
            "TMPDIR": workdir,
            "TEMP": workdir,
            "TMP": workdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if sys.platform == "win32":
            # Windows needs these for the interpreter to start at all.
            for key in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT"):
                if key in os.environ:
                    env[key] = os.environ[key]

        try:
            process = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-I",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
                # A new process group so a timeout kills grandchildren too.
                start_new_session=sys.platform != "win32",
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(self._wrap(code).encode("utf-8")),
                    timeout=limit,
                )
            except asyncio.TimeoutError:
                await self._kill_tree(process)
                return SandboxResult(
                    stderr=f"Execution exceeded the {limit:.0f}s time limit.",
                    exit_code=124,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    timed_out=True,
                )

            out, out_truncated = self._cap(stdout.decode("utf-8", errors="replace"))
            err, err_truncated = self._cap(stderr.decode("utf-8", errors="replace"))
            return SandboxResult(
                stdout=out,
                stderr=err,
                exit_code=process.returncode if process.returncode is not None else -1,
                duration_ms=(time.perf_counter() - started) * 1000,
                truncated=out_truncated or err_truncated,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _cap(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        return text[: self.max_output_chars] + "\n...[truncated]", True

    @staticmethod
    async def _kill_tree(process: asyncio.subprocess.Process) -> None:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), 9)
            else:
                process.kill()
            await process.wait()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
                await process.wait()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass

    async def health(self) -> tuple[str, str | None]:
        return "degraded", (
            "subprocess mode provides no isolation - code runs on the host with "
            "the server's privileges"
        )
