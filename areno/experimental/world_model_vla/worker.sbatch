#!/usr/bin/env bash
set -euo pipefail
unset GREP_OPTIONS || true

ARENO_PYTHON=$1
SETUP_SCRIPT=$2
shift 2

if [[ "${SETUP_SCRIPT}" != "-" ]]; then
    source "${SETUP_SCRIPT}"
fi

ulimit -l unlimited 2>/dev/null || true
ulimit -n 65535 2>/dev/null || true
ulimit -u 4125556 2>/dev/null || true

exec "${ARENO_PYTHON}" -m areno.experimental.world_model_vla "$@"
