"""Environment capture and GPU/vLLM sampling for the concurrency experiment.

Two jobs, both of which exist so a result can be defended rather than just
reported:

* ``capture_environment`` fingerprints everything that could change a number -
  model revision, vLLM version, driver, engine flags, git commit. A run whose
  fingerprint differs from another is not comparable to it.
* ``GpuSampler`` polls vLLM's Prometheus endpoint and ``nvidia-smi`` on a
  background thread while a run is in flight, so GPU utilisation and KV-cache
  pressure are recorded *during* the measured window rather than guessed after.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Gauges worth having per run. Counters are differenced across the window;
# gauges are summarised as mean/max.
_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
_COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def _fetch(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", "replace")


def parse_prometheus(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Pull the named metrics out of a Prometheus exposition payload."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for name in names:
            if line.startswith(name):
                match = re.search(r"\s([0-9eE+.\-]+)$", line.strip())
                if match:
                    try:
                        found[name] = float(match.group(1))
                    except ValueError:
                        pass
    return found


def _nvidia_smi() -> dict[str, float]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        util, used, total, temp = (float(x.strip()) for x in out.split(","))
        return {"gpu_util_pct": util, "gpu_mem_used_mib": used,
                "gpu_mem_total_mib": total, "gpu_temp_c": temp}
    except Exception:
        return {}


# --- environment ------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def capture_environment(vllm_base: str, backend_base: str) -> dict[str, Any]:
    """Fingerprint everything that must be constant across conditions."""
    env: dict[str, Any] = {"captured_at": time.time()}

    env["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    env["git_dirty"] = bool(_run(["git", "status", "--porcelain"]))

    try:
        models = json.loads(_fetch(f"{vllm_base}/models"))
        entry = models["data"][0]
        env["model"] = {"id": entry.get("id"), "max_model_len": entry.get("max_model_len")}
    except Exception as exc:
        env["model"] = {"error": str(exc)}

    try:
        env["backend_config"] = json.loads(_fetch(f"{backend_base}/config"))
    except Exception as exc:
        env["backend_config"] = {"error": str(exc)}

    env["gpu"] = _nvidia_smi()
    env["nvidia_smi_version"] = _run(["nvidia-smi", "--query-gpu=driver_version",
                                      "--format=csv,noheader"])
    env["vllm_version"] = _run(["python", "-c",
                                "import vllm;print(vllm.__version__)"])
    # The engine's own flags: the authoritative record of how it was launched.
    env["vllm_argv"] = _run(["bash", "-lc",
                             "cat /proc/$(pgrep -f 'vllm serve' | head -1)/cmdline "
                             "2>/dev/null | tr '\\0' ' '"])
    return env


# --- sampling ---------------------------------------------------------------


@dataclass
class GpuSample:
    t: float
    values: dict[str, float] = field(default_factory=dict)


class GpuSampler:
    """Polls vLLM and nvidia-smi on a thread for the duration of a run."""

    def __init__(self, vllm_base: str, interval_s: float = 0.25) -> None:
        self.metrics_url = f"{vllm_base.rsplit('/v1', 1)[0]}/metrics"
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first_counters: dict[str, float] = {}
        self._last_counters: dict[str, float] = {}

    def _poll(self) -> None:
        while not self._stop.is_set():
            values: dict[str, float] = {}
            try:
                text = _fetch(self.metrics_url, timeout=3)
                values.update(parse_prometheus(text, _GAUGES))
                counters = parse_prometheus(text, _COUNTERS)
                if counters:
                    if not self._first_counters:
                        self._first_counters = dict(counters)
                    self._last_counters = dict(counters)
            except Exception:
                pass
            values.update(_nvidia_smi())
            if values:
                self.samples.append(GpuSample(time.time(), values))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuSampler:
        self.samples.clear()
        self._first_counters.clear()
        self._last_counters.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": len(self.samples)}
        if not self.samples:
            return out
        for key in set().union(*(s.values.keys() for s in self.samples)):
            series = [s.values[key] for s in self.samples if key in s.values]
            if series:
                out[key] = {
                    "mean": round(sum(series) / len(series), 4),
                    "max": round(max(series), 4),
                }
        # Counter deltas over the measured window. The prefix-cache pair is the
        # confound that could manufacture a parallel win, so it is recorded for
        # every single run rather than sampled.
        queries = self._last_counters.get("vllm:prefix_cache_queries_total", 0.0) - \
            self._first_counters.get("vllm:prefix_cache_queries_total", 0.0)
        hits = self._last_counters.get("vllm:prefix_cache_hits_total", 0.0) - \
            self._first_counters.get("vllm:prefix_cache_hits_total", 0.0)
        out["prefix_cache"] = {
            "queried_tokens": queries,
            "hit_tokens": hits,
            "hit_rate": round(hits / queries, 4) if queries > 0 else None,
        }
        return out


def wait_until_drained(vllm_base: str, timeout_s: float = 60.0) -> bool:
    """Block until vLLM reports nothing running or waiting.

    A batching scheduler will happily mix requests from two adjacent runs; a
    condition measured while the previous one is still draining is not that
    condition.
    """
    url = f"{vllm_base.rsplit('/v1', 1)[0]}/metrics"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            values = parse_prometheus(_fetch(url, timeout=3), _GAUGES)
            running = values.get("vllm:num_requests_running", 0.0)
            waiting = values.get("vllm:num_requests_waiting", 0.0)
            if running == 0 and waiting == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
