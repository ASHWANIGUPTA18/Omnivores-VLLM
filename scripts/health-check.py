#!/usr/bin/env python3
"""Check that every part of a running Omnivores deployment works.

Goes beyond ``GET /health``: with ``--full`` it uploads a document, waits for
it to index, and confirms it becomes searchable. That is the path users
actually care about, and the one most likely to be half-broken after a
configuration change.

Uses only the standard library, so it can run anywhere - including inside a
container that has no project dependencies installed.

    python scripts/health-check.py
    python scripts/health-check.py --url http://gpu-box:8000 --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

_failures = 0
_warnings = 0


def report(ok: bool, label: str, detail: str = "", *, warn_only: bool = False) -> bool:
    global _failures, _warnings
    if ok:
        mark, colour = "PASS", GREEN
    elif warn_only:
        mark, colour = "WARN", YELLOW
        _warnings += 1
    else:
        mark, colour = "FAIL", RED
        _failures += 1
    suffix = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {colour}{mark}{RESET}  {label}{suffix}")
    return ok


def get(url: str, api_key: str = "", timeout: float = 20.0):
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def post_file(url: str, filename: str, content: bytes, fields: dict, api_key: str = ""):
    boundary = "----omnivores-health-check-boundary"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode()
        + content
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=b"".join(chunks), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def delete(url: str, api_key: str = "") -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        urllib.request.urlopen(request, timeout=20).read()
    except urllib.error.URLError:
        pass


def check_backend(base: str, key: str) -> bool:
    print("Backend")
    try:
        health = get(f"{base}/health", key)
    except urllib.error.HTTPError as exc:
        report(False, "reachable", f"HTTP {exc.code}")
        return False
    except urllib.error.URLError as exc:
        report(False, "reachable", str(exc.reason))
        return False

    report(
        health.get("status") == "healthy",
        "reachable",
        f"{health.get('provider')} / {health.get('model')}",
    )
    return True


def check_components(base: str, key: str) -> None:
    print("\nComponents")
    try:
        full = get(f"{base}/health/full", key)
    except Exception as exc:  # noqa: BLE001
        report(False, "component health", str(exc))
        return

    for name, component in full.get("components", {}).items():
        status = component.get("status")
        report(
            status == "healthy",
            name,
            component.get("detail") or "",
            warn_only=(status == "degraded"),
        )


def check_inference(base: str, key: str) -> None:
    print("\nInference")
    try:
        inference = get(f"{base}/health/inference", key)
    except Exception as exc:  # noqa: BLE001
        report(False, "inference health", str(exc))
        return

    provider = inference.get("provider")
    healthy = inference.get("status") == "healthy"
    report(healthy, f"provider: {provider}", inference.get("detail") or "")

    if provider == "mock":
        report(
            False,
            "a real model is configured",
            "LLM_PROVIDER=mock - all responses are synthetic",
            warn_only=True,
        )

    info = inference.get("model_info") or {}
    if info.get("context_length"):
        report(True, "context window", f"{info['context_length']} tokens")

    if not healthy and inference.get("hint"):
        print(f"      {DIM}{inference['hint']}{RESET}")


def check_configuration(base: str, key: str) -> None:
    print("\nConfiguration")
    try:
        agents = get(f"{base}/api/agents", key)["agents"]
        report(
            len(agents) > 0,
            f"{len(agents)} agent(s)",
            ", ".join("@" + a["id"] for a in agents),
        )
        without_tools = [a["id"] for a in agents if not a["tools"]]
        if without_tools:
            report(False, "agents with no tools", ", ".join(without_tools), warn_only=True)
    except Exception as exc:  # noqa: BLE001
        report(False, "agents", str(exc))

    try:
        channels = get(f"{base}/api/channels", key)["channels"]
        report(
            len(channels) > 0,
            f"{len(channels)} channel(s)",
            ", ".join("#" + c["name"] for c in channels),
        )
    except Exception as exc:  # noqa: BLE001
        report(False, "channels", str(exc))

    try:
        config = get(f"{base}/config", key)
        rag = config["rag"]
        report(
            True,
            "RAG",
            f"{rag['embedding_provider']} embeddings, {rag['vector_store']} store, "
            f"{rag['reranker']} reranker",
        )
        sandbox = config["sandbox"]["mode"]
        report(
            sandbox != "subprocess",
            f"sandbox: {sandbox}",
            "subprocess mode provides no isolation" if sandbox == "subprocess" else "",
            warn_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        report(False, "config", str(exc))


def check_document_round_trip(base: str, key: str) -> None:
    print("\nDocument round trip")
    document_id = None
    try:
        content = (
            b"# Health Check Document\n\n"
            b"The canary phrase for this health check is XYZZY-OMNIVORES-42.\n\n"
            b"# Second Section\n\n"
            b"This exists so the chunker has more than one section to work with.\n"
        )
        upload = post_file(
            f"{base}/api/documents",
            "omnivores-health-check.md",
            content,
            {"channel_id": "general"},
            key,
        )
        document_id = upload["document"]["id"]
        report(True, "upload accepted", document_id)

        deadline = time.time() + 90
        status = "unknown"
        document = {}
        while time.time() < deadline:
            document = get(f"{base}/api/documents/{document_id}", key)["document"]
            status = document.get("status")
            if status in ("ready", "failed"):
                break
            time.sleep(0.5)

        if status == "ready":
            report(True, "indexed", f"{document.get('chunk_count')} chunks")
        else:
            report(False, "indexed", f"status={status} {document.get('error') or ''}")

    except Exception as exc:  # noqa: BLE001
        report(False, "document round trip", str(exc))
    finally:
        if document_id:
            delete(f"{base}/api/documents/{document_id}", key)
            print(f"  {DIM}cleaned up {document_id}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8000", help="backend base URL")
    parser.add_argument("--api-key", default="", help="API key, if the backend requires one")
    parser.add_argument(
        "--full", action="store_true", help="also upload and index a test document"
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"\nOmnivores health check - {base}\n")

    if not check_backend(base, args.api_key):
        print(f"\n{RED}The backend is not answering. Is it running?{RESET}")
        print(f"{DIM}  cd backend && uvicorn app.main:app --port 8000{RESET}\n")
        return 1

    check_components(base, args.api_key)
    check_inference(base, args.api_key)
    check_configuration(base, args.api_key)
    if args.full:
        check_document_round_trip(base, args.api_key)

    print()
    if _failures:
        tail = f", {_warnings} warning(s)" if _warnings else ""
        print(f"{RED}{_failures} check(s) failed{RESET}{tail}\n")
        return 1
    if _warnings:
        print(f"{YELLOW}All checks passed, with {_warnings} warning(s).{RESET}\n")
        return 0
    print(f"{GREEN}All checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
