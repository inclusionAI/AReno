#!/usr/bin/env bash
set -euo pipefail
# This socket belongs to the nested daemon. Never bind-mount a host Docker socket.
if [ -S /var/run/docker.sock ]; then
    echo 'Refusing an existing Docker socket; start a fresh DinD container.' >&2
    exit 1
fi
export DOCKER_HOST=unix:///var/run/docker.sock
unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH
dockerd --host="$DOCKER_HOST" --data-root=/var/lib/areno-docker \
    --label=areno.pi.dind=true > /var/log/areno-dockerd.log 2>&1 &
daemon_pid=$!
child_pid=''
cleanup() {
    if [ -n "$child_pid" ]; then kill -TERM "$child_pid" 2>/dev/null || true; fi
    kill -TERM "$daemon_pid" 2>/dev/null || true
    wait "$daemon_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ready=0
for _ in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$daemon_pid" 2>/dev/null || break
    sleep 1
done
if [ "$ready" -ne 1 ]; then cat /var/log/areno-dockerd.log >&2; exit 1; fi
"$@" <&0 &
child_pid=$!
wait "$child_pid"
