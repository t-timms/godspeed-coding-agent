#!/usr/bin/env bash
# Run ONE agent shell command in a private mount + PID namespace (Linux, unprivileged).
#
# Install as Godspeed's shell for a benchmark run:
#     export GODSPEED_SHELL_WRAPPER=/abs/path/to/scripts/agent_shell_isolate.sh
# Godspeed then calls   agent_shell_isolate.sh -c "<command>"   with cwd = the task workspace.
#
# Why: a benchmark agent's shell shares the host filesystem, and agents do go looking. In one run a
# model searched the host for hidden tests and gold data (`find / -name "*1359*"`, greps of a failing
# test's name over /home). Here each command sees: the task workspace, its venv, a per-task private
# /tmp, system directories, and nothing of $HOME or /mnt; all capabilities are dropped, so it cannot
# undo the mounts. The runner itself is NOT confined.
#
# Environment:
#   AGENT_PRIV      host dir that becomes the agent's /tmp (persists across commands; use one per task).
#                   Default: a fresh temp dir per command (scratch files do not survive the command).
#   AGENT_VENV      host path of the task venv; visible at the same path (optional).
#   AGENT_KEEP_RO   ':'-separated host paths bound READ-ONLY at the same path even when under a masked
#                   directory. Default: $AGENT_REAL_HOME/.local/share/uv (uv-managed Pythons).
#   AGENT_MASK      ':'-separated directories replaced by an empty tmpfs. Default: $AGENT_REAL_HOME:/mnt
#   AGENT_REAL_HOME home directory to hide. Default: $HOME.
#
# DNS: on WSL2 /etc/resolv.conf points into /mnt/wsl; the resolver file is re-bound read-only after /mnt
# is masked, otherwise the agent silently loses the network (pip installs fail with "no matching
# distribution"). Fails closed: if namespaces are unavailable the command is NOT run unsandboxed.
set -u

[ "${1:-}" = "-c" ] && [ $# -ge 2 ] || { echo "usage: $0 -c COMMAND" >&2; exit 2; }
command -v unshare >/dev/null 2>&1 && command -v setpriv >/dev/null 2>&1 || {
  echo "agent_shell_isolate: needs util-linux unshare and setpriv" >&2; exit 97; }

REAL_HOME="${AGENT_REAL_HOME:-$HOME}"
export AGENT_CMD="$2" WS="$(pwd -P)" REAL_HOME
export KEEP_RO="${AGENT_KEEP_RO-$REAL_HOME/.local/share/uv}"
export MASK="${AGENT_MASK-$REAL_HOME:/mnt}"
export RESOLV="$(readlink -f /etc/resolv.conf 2>/dev/null || true)"
export STAGE="/dev/shm/agentk_$$"
export AGENT_VENV="${AGENT_VENV:-}"
if [ -z "${AGENT_PRIV:-}" ]; then AGENT_PRIV="$(mktemp -d /var/tmp/agent_priv.XXXXXX)"; fi
mkdir -p "$AGENT_PRIV"
export AGENT_PRIV

exec unshare --user --map-root-user --mount --pid --fork --kill-child --mount-proc \
  --propagation private /bin/bash -c '
set -e
mkdir -p "$STAGE/ws" "$STAGE/priv" "$STAGE/venv"
mount --bind "$WS" "$STAGE/ws"
mount --bind "$AGENT_PRIV" "$STAGE/priv"
have_venv=""
if [ -n "$AGENT_VENV" ] && [ -d "$AGENT_VENV" ]; then
  mount --bind "$AGENT_VENV" "$STAGE/venv"; have_venv=1
fi

# read-only keeps, staged before their parents are masked
i=0; keeps=()
IFS=: read -r -a keep_paths <<< "$KEEP_RO"
for k in "${keep_paths[@]}"; do
  [ -n "$k" ] && [ -e "$k" ] || continue
  i=$((i+1)); mkdir -p "$STAGE/keep$i"
  mount --bind "$k" "$STAGE/keep$i"; mount -o remount,bind,ro "$STAGE/keep$i"
  keeps+=("$i:$k")
done
if [ -n "$RESOLV" ] && [ -e "$RESOLV" ]; then
  touch "$STAGE/resolv"; mount --bind "$RESOLV" "$STAGE/resolv"
fi

# hide the masked directories
IFS=: read -r -a mask_paths <<< "$MASK"
for m in "${mask_paths[@]}"; do
  [ -n "$m" ] && [ -d "$m" ] && mount -t tmpfs -o mode=755 tmpfs "$m"
done

# private /tmp, then put the workspace, venv and keeps back
mount --bind "$STAGE/priv" /tmp
mkdir -p "$WS" /tmp/agent_home
mount --bind "$STAGE/ws" "$WS"
# decided while staging: after /tmp is replaced the original path no longer exists to test
if [ -n "$have_venv" ]; then
  mkdir -p "$AGENT_VENV"; mount --bind "$STAGE/venv" "$AGENT_VENV"
fi
for entry in "${keeps[@]}"; do
  n="${entry%%:*}"; k="${entry#*:}"
  mkdir -p "$k"; mount --bind "$STAGE/keep$n" "$k"
done
if [ -n "$RESOLV" ] && [ -e "$STAGE/resolv" ]; then
  mkdir -p "$(dirname "$RESOLV")"; touch "$RESOLV"
  mount --bind "$STAGE/resolv" "$RESOLV"; mount -o remount,bind,ro "$RESOLV"
fi

export HOME=/tmp/agent_home
cd "$WS"
exec setpriv --bounding-set=-all --inh-caps=-all --no-new-privs /bin/bash -c "$AGENT_CMD"
'
