"""Docker-backed execution sandbox.

We shell out to the ``docker`` CLI rather than using the SDK. That is
deliberate: the SDK would mean either mounting the Docker socket into the
backend container (a full host compromise if the backend is ever exploited) or
adding a dependency for something a subprocess call does fine.

Every container is created with:

* ``--network none``          - no egress, no lateral movement
* ``--read-only``             - immutable root filesystem
* ``--tmpfs /tmp``            - a small writable scratch area, noexec
* ``--cap-drop ALL``          - no capabilities
* ``--security-opt no-new-privileges``
* ``--pids-limit``            - fork bombs cannot exhaust host PIDs
* ``--memory`` / ``--cpus``   - resource caps
* ``--user 65534:65534``      - nobody, never root
* ``--rm``                    - no accumulating dead containers

The code is passed on stdin, never interpolated into a shell command, so there
is no shell-injection surface between us and the interpreter.
"""

from __future__ import annotations

import asyncio
import shutil
import time

from app.core.config import settings
from app.core.exceptions import SandboxUnavailable
from app.core.logging import get_logger
from app.sandbox.base import ExecutionSandbox, SandboxResult

logger = get_logger(__name__)


class DockerSandbox(ExecutionSandbox):
    name = "docker"

    def __init__(
        self,
        *,
        image: str | None = None,
        timeout: float | None = None,
        memory_limit_mb: int | None = None,
        cpu_limit: float | None = None,
        max_output_chars: int | None = None,
    ) -> None:
        self.image = image or settings.python_sandbox_image
        self.timeout = timeout or settings.python_timeout_seconds
        self.memory_limit_mb = memory_limit_mb or settings.python_memory_limit_mb
        self.cpu_limit = cpu_limit or settings.python_cpu_limit
        self.max_output_chars = max_output_chars or settings.python_max_output_chars
        self._docker_path: str | None = None

    def _docker(self) -> str:
        if self._docker_path is None:
            found = shutil.which("docker")
            if not found:
                raise SandboxUnavailable(
                    "the docker CLI is not on PATH",
                    user_message=(
                        "Code execution is unavailable: Docker is not installed on "
                        "the server."
                    ),
                )
            self._docker_path = found
        return self._docker_path

    def _command(self, timeout: float) -> list[str]:
        return [
            self._docker(),
            "run",
            "--rm",
            "--interactive",
            "--network", "none",
            "--read-only",
            # noexec stops the code writing a binary to /tmp and running it.
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "64",
            "--memory", f"{self.memory_limit_mb}m",
            # Without this, the kernel can swap instead of enforcing the cap.
            "--memory-swap", f"{self.memory_limit_mb}m",
            "--cpus", str(self.cpu_limit),
            "--user", "65534:65534",
            "--workdir", "/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
            "--env", "HOME=/tmp",
            self.image,
            "python", "-I", "-",
        ]

    async def run_python(self, code: str, *, timeout: float | None = None) -> SandboxResult:
        limit = timeout or self.timeout
        started = time.perf_counter()

        try:
            process = await asyncio.create_subprocess_exec(
                *self._command(limit),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailable(f"cannot start docker: {exc}") from exc
        except OSError as exc:
            raise SandboxUnavailable(f"cannot start sandbox container: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(code.encode("utf-8")),
                # A little headroom over the in-container limit so we normally
                # see the container's own exit rather than killing it blind.
                timeout=limit + 5.0,
            )
        except asyncio.TimeoutError:
            await self._terminate(process)
            return SandboxResult(
                stderr=f"Execution exceeded the {limit:.0f}s time limit.",
                exit_code=124,
                duration_ms=(time.perf_counter() - started) * 1000,
                timed_out=True,
            )

        duration_ms = (time.perf_counter() - started) * 1000
        out, out_truncated = self._cap(stdout.decode("utf-8", errors="replace"))
        err, err_truncated = self._cap(stderr.decode("utf-8", errors="replace"))

        if process.returncode == 137:
            # OOM-killed by the cgroup limit.
            err = (
                err + f"\nThe process was killed for exceeding the "
                f"{self.memory_limit_mb} MB memory limit."
            ).strip()

        return SandboxResult(
            stdout=out,
            stderr=err,
            exit_code=process.returncode if process.returncode is not None else -1,
            duration_ms=duration_ms,
            truncated=out_truncated or err_truncated,
        )

    def _cap(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        return text[: self.max_output_chars] + "\n...[truncated]", True

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    async def health(self) -> tuple[str, str | None]:
        try:
            docker = self._docker()
        except SandboxUnavailable as exc:
            return "unavailable", exc.message

        try:
            process = await asyncio.create_subprocess_exec(
                docker, "image", "inspect", self.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return "unavailable", f"docker not responding: {exc}"

        if process.returncode != 0:
            return "unavailable", (
                f"sandbox image '{self.image}' is not built. Run "
                "`docker build -t omnivores-sandbox:latest backend/sandbox-image`."
            )
        return "healthy", f"image {self.image}"
