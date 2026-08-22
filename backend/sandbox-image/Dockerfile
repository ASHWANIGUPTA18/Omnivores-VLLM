# The Python sandbox image.
#
# This is what model-generated code runs inside. It is deliberately minimal:
# every package here is something untrusted code can reach for.
#
# Build:
#   docker build -t omnivores-sandbox:latest backend/sandbox-image
#
# The container is launched by app/sandbox/docker_sandbox.py with no network,
# a read-only root filesystem, dropped capabilities, and memory/CPU/PID caps.
# Those runtime flags are the security boundary; this image only decides what
# code is available inside it.

FROM python:3.11-slim

# The standard library covers most of what agents need. Uncomment to widen the
# sandbox, understanding that each addition is more surface area:
#
# RUN pip install --no-cache-dir numpy pandas
#
# Do NOT add requests, httpx, or any network client. The container has no
# network, so they would only produce confusing errors.

# nobody. The runtime also passes --user, so this is defence in depth.
USER 65534:65534

WORKDIR /tmp

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# Code arrives on stdin. -I is isolated mode: no user site-packages, no
# PYTHONPATH, no current directory on sys.path.
CMD ["python", "-I", "-"]
