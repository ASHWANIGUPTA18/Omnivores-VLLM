#!/usr/bin/env bash
# Omnivores AI - one-shot setup and launcher for Ubuntu/Debian with an NVIDIA GPU.
#
#   ./omnivores.sh            install what is missing, then start and open the UI
#   ./omnivores.sh start      start (skip the install checks)
#   ./omnivores.sh stop       stop the stack
#   ./omnivores.sh logs       follow logs
#   ./omnivores.sh status     show what is running
#   ./omnivores.sh shortcut   put a double-clickable icon on the Desktop
#   ./omnivores.sh --yes      don't ask before installing system packages
#
# Safe to re-run: every step checks before acting.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $ROOT/docker-compose.gpu.yml"
ASSUME_YES=0
ACTION="setup"

for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    start|stop|logs|status|shortcut|setup) ACTION="$arg" ;;
    -h|--help) awk 'NR>1 && /^#/{sub(/^# ?/,"");print;next} NR>1{exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

# --- output helpers ---------------------------------------------------------
B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
note() { printf '    %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n    %s✗ %s%s\n\n' "$R" "$*" "$N"; exit 1; }

confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  printf '\n    %s%s%s\n    Continue? [y/N] ' "$B" "$1" "$N"
  read -r reply </dev/tty
  case "$reply" in [yY]*) return 0 ;; *) die "cancelled by user" ;; esac
}

# docker may need the group that this shell has not picked up yet.
d() { if [ "${NEED_SG:-0}" = 1 ]; then sg docker -c "$*"; else eval "$*"; fi; }

# --- lightweight actions ----------------------------------------------------
case "$ACTION" in
  stop)   step "Stopping"; d "$COMPOSE down"; ok "stopped"; exit 0 ;;
  start)
    step "Starting (skipping install checks)"
    d "$COMPOSE up -d" || die "could not start - run ./omnivores.sh with no arguments for the full setup"
    printf '    waiting for the backend'
    for _ in $(seq 1 90); do
      curl -fsS http://localhost:8000/health >/dev/null 2>&1 && break
      printf '.'; sleep 2
    done
    echo
    curl -fsS http://localhost:8000/health >/dev/null 2>&1 || die "backend did not answer. Check: ./omnivores.sh logs"
    ok "ready - http://localhost:3000"
    command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 >/dev/null 2>&1 &
    exit 0 ;;
  logs)   d "$COMPOSE logs -f"; exit 0 ;;
  status) d "$COMPOSE ps"; exit 0 ;;
  shortcut)
    DESK="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
    mkdir -p "$DESK"
    cat > "$DESK/Omnivores AI.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Omnivores AI
Comment=Start the local AI workspace
Exec=bash -c 'cd "$ROOT" && ./omnivores.sh; echo; read -rp "Press Enter to close..."'
Path=$ROOT
Terminal=true
Icon=applications-science
Categories=Development;
EOF
    chmod +x "$DESK/Omnivores AI.desktop"
    gio set "$DESK/Omnivores AI.desktop" metadata::trusted true 2>/dev/null || true
    ok "created: $DESK/Omnivores AI.desktop"
    note "GNOME may ask you to 'Allow Launching' the first time (right-click the icon)."
    exit 0 ;;
esac

printf '%s\n  Omnivores AI\n%s\n' "$B" "$N"

# --- 1. platform ------------------------------------------------------------
step "Checking the machine"
[ -r /etc/os-release ] || die "not a Linux system this script understands"
. /etc/os-release
case "${ID:-}${ID_LIKE:-}" in
  *ubuntu*|*debian*) ok "OS: ${PRETTY_NAME}" ;;
  *) die "this installer handles Ubuntu/Debian only. You have: ${PRETTY_NAME:-unknown}. Follow docs/RUNBOOK-GPU.md by hand." ;;
esac

command -v nvidia-smi >/dev/null 2>&1 \
  || die "no NVIDIA driver found. Install one first (Ubuntu: Software & Updates -> Additional Drivers), reboot, then re-run this."
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
[ "${DRIVER%%.*}" -ge 525 ] 2>/dev/null || die "driver $DRIVER is too old; vLLM needs 525 or newer"
ok "GPU: $GPU (${VRAM} MiB), driver $DRIVER"
[ "${VRAM:-0}" -ge 22000 ] 2>/dev/null || note "under 22 GB of VRAM - if vLLM runs out of memory, set MAX_MODEL_LEN=16384 in .env"

AVAIL=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${AVAIL:-999}" -ge 60 ] 2>/dev/null || note "only ${AVAIL} GB free - the model and images need roughly 60-100 GB"

# --- 2. docker --------------------------------------------------------------
step "Docker"
if command -v docker >/dev/null 2>&1; then
  ok "already installed ($(docker --version))"
else
  confirm "Docker is not installed. This will run the official get.docker.com installer with sudo."
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh || die "could not download the Docker installer"
  sudo sh /tmp/get-docker.sh || die "Docker installation failed"
  ok "Docker installed"
fi

docker compose version >/dev/null 2>&1 || die "'docker compose' v2 is missing. Install docker-compose-plugin and re-run."
ok "compose v2 available"

NEED_SG=0
if ! docker info >/dev/null 2>&1; then
  if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    NEED_SG=1
    note "you are in the 'docker' group but this shell predates it - using 'sg docker' for now"
  else
    confirm "Your user cannot use Docker. This will add '$USER' to the 'docker' group with sudo."
    sudo usermod -aG docker "$USER" || die "could not add you to the docker group"
    NEED_SG=1
    ok "added $USER to the docker group"
    note "log out and back in later so this sticks permanently"
  fi
  d "docker info" >/dev/null 2>&1 || die "still cannot reach the Docker daemon. Log out, log back in, and re-run."
fi
ok "Docker daemon reachable"

# --- 3. GPU inside containers ----------------------------------------------
step "NVIDIA Container Toolkit"
if d "docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi" >/dev/null 2>&1; then
  ok "containers can see the GPU"
else
  confirm "Containers cannot see the GPU. This will install nvidia-container-toolkit with sudo and restart Docker."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || die "could not add the NVIDIA key"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq || die "apt update failed"
  sudo apt-get install -y -qq nvidia-container-toolkit || die "could not install nvidia-container-toolkit"
  sudo nvidia-ctk runtime configure --runtime=docker || die "nvidia-ctk configure failed"
  sudo systemctl restart docker || die "could not restart Docker"
  sleep 3
  d "docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi" >/dev/null 2>&1 \
    || die "containers still cannot see the GPU. See docs/RUNBOOK-GPU.md section 1."
  ok "installed - containers can see the GPU"
fi

# --- 4. configuration -------------------------------------------------------
step "Configuration"
if [ -f "$ROOT/.env" ]; then
  ok ".env already present (leaving it alone)"
else
  cp "$ROOT/.env.example" "$ROOT/.env" || die "could not create .env"
  ok "created .env from .env.example"
fi
MODEL=$(grep -E '^MODEL_NAME=' "$ROOT/.env" | cut -d= -f2-)
PARSER=$(grep -E '^TOOL_CALL_PARSER=' "$ROOT/.env" | cut -d= -f2-)
[ -n "$MODEL" ]  || die "MODEL_NAME is empty in .env"
[ -n "$PARSER" ] || die "TOOL_CALL_PARSER is empty in .env"
grep -qE '^API_KEY=.+' "$ROOT/.env" && die "API_KEY is set in .env. The frontend bakes its key in at build time and this stack passes none, so the UI would 401 on everything. Empty it and re-run."
ok "model: $MODEL   parser: $PARSER"

# --- 5. build + verify the parser ------------------------------------------
step "Building the model server (first time: several minutes)"
d "$COMPOSE build vllm" || die "could not build the vLLM image"
ok "image built"

step "Verifying tool-calling support"
if d "docker run --rm --gpus all --entrypoint vllm omnivores-vllm:latest serve --help" 2>/dev/null | grep -q -- "$PARSER"; then
  ok "parser '$PARSER' is supported by this vLLM build"
else
  note "parser '$PARSER' was NOT found in this vLLM build."
  note "Agents will answer but will never call tools - no document search, no citations."
  note "Fix: pin VLLM_VERSION in .env to a release that has it, then re-run."
  confirm "Continue anyway with tool calling likely broken?"
fi

# --- 6. run -----------------------------------------------------------------
step "Starting (first run downloads the model, ~16 GB - this is the slow part)"
d "$COMPOSE up -d --build" || die "could not start the stack"

printf '    waiting for the model server'
for i in $(seq 1 180); do
  state=$(d "$COMPOSE ps --format '{{.Service}} {{.Health}}'" 2>/dev/null | awk '$1=="vllm"{print $2}')
  [ "$state" = "healthy" ] && break
  if [ "$((i % 15))" = 0 ]; then
    printf '\n    still loading (%d min). Watch progress: ./omnivores.sh logs\n    ' "$((i/6))"
  else
    printf '.'
  fi
  sleep 10
done
printf '\n'
[ "$state" = "healthy" ] || die "the model server did not come up in 30 minutes. Check: ./omnivores.sh logs"
ok "model server healthy"

printf '    waiting for the app'
for i in $(seq 1 60); do
  curl -fsS http://localhost:8000/health >/dev/null 2>&1 && break
  printf '.'; sleep 2
done
printf '\n'
curl -fsS http://localhost:8000/health >/dev/null 2>&1 || die "the backend did not answer. Check: ./omnivores.sh logs"
ok "backend healthy"

# --- 7. verify --------------------------------------------------------------
step "Checking the full pipeline"
if command -v python3 >/dev/null 2>&1 && [ -f "$ROOT/scripts/health-check.py" ]; then
  python3 "$ROOT/scripts/health-check.py" --full || note "health check reported problems - the UI may still work"
else
  note "python3 not found, skipping the deeper check"
fi

# --- 8. done ----------------------------------------------------------------
printf '\n%s  Ready.%s  http://localhost:3000\n\n' "$G" "$N"
cat <<'TIPS'
  First thing to try, to prove tool calling works:

      @coder compute 47**13

  A tool strip must appear above the answer. If it does not, the parser is
  wrong and answers will be ungrounded - see docs/RUNBOOK-GPU.md step 3.

  Then upload a PDF in the sidebar, wait for "ready", and ask:

      @hr what does this document say about the notice period?

  Note: the first upload pauses while ~90 MB of embedding model downloads.

    ./omnivores.sh logs     follow logs
    ./omnivores.sh stop     shut it down
    ./omnivores.sh          start it again

TIPS

command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 >/dev/null 2>&1 &
exit 0
