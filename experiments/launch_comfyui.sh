#!/usr/bin/env bash
#
# Launch multiple ComfyUI instances on consecutive ports.
#
# Usage:
#   ./experiments/launch_comfyui.sh              # 1 instance on :8188
#   ./experiments/launch_comfyui.sh 3            # 3 instances on :8188, :8189, :8190
#   ./experiments/launch_comfyui.sh 3 8200       # 3 instances on :8200, :8201, :8202
#   COMFYUI_DIR=/path/to/ComfyUI ./experiments/launch_comfyui.sh 2
#
# Prints the COMFYUI_ADDRS= line to use with run_benchmark.py.

set -euo pipefail

N_INSTANCES="${1:-1}"
BASE_PORT="${2:-8188}"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
COMFYUI_MAIN="${COMFYUI_DIR}/main.py"

COMFYUI_PYTHON="${COMFYUI_PYTHON:-}"
if [[ -z "$COMFYUI_PYTHON" ]]; then
    if [[ -x /venv/comfyui/bin/python3.12 ]]; then
        COMFYUI_PYTHON=/venv/comfyui/bin/python3.12
    else
        COMFYUI_PYTHON=python
    fi
fi

if [[ ! -f "$COMFYUI_MAIN" ]]; then
    echo "ERROR: ComfyUI not found at $COMFYUI_MAIN" >&2
    echo "Set COMFYUI_DIR to your ComfyUI installation." >&2
    exit 1
fi

PIDS=()
ADDRS=()

cleanup() {
    echo ""
    echo "Shutting down ComfyUI instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  Stopped PID $pid" || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

for (( i=0; i<N_INSTANCES; i++ )); do
    PORT=$(( BASE_PORT + i ))
    ADDR="127.0.0.1:${PORT}"
    ADDRS+=("$ADDR")

    echo "Starting ComfyUI instance $((i+1))/${N_INSTANCES} on port ${PORT}..."
    "$COMFYUI_PYTHON" "$COMFYUI_MAIN" --port "$PORT" --listen 127.0.0.1 \
        > "/tmp/comfyui_${PORT}.log" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}  Log: /tmp/comfyui_${PORT}.log"
done

ADDRS_STR=$(IFS=,; echo "${ADDRS[*]}")
echo ""
echo "============================================================"
echo "All ${N_INSTANCES} ComfyUI instance(s) launching."
echo ""
echo "  export COMFYUI_ADDRS=${ADDRS_STR}"
echo ""
echo "Waiting for instances to be ready..."
echo "============================================================"

for (( i=0; i<N_INSTANCES; i++ )); do
    PORT=$(( BASE_PORT + i ))
    ADDR="127.0.0.1:${PORT}"
    for (( attempt=1; attempt<=120; attempt++ )); do
        if curl -s "http://${ADDR}/system_stats" > /dev/null 2>&1; then
            echo "  Instance $((i+1)) (port ${PORT}) ready."
            break
        fi
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "ERROR: Instance $((i+1)) (port ${PORT}) exited unexpectedly." >&2
            echo "Check /tmp/comfyui_${PORT}.log for details." >&2
            exit 1
        fi
        sleep 2
    done
    if ! curl -s "http://${ADDR}/system_stats" > /dev/null 2>&1; then
        echo "ERROR: Instance $((i+1)) (port ${PORT}) did not become ready in 240s." >&2
        exit 1
    fi
done

echo ""
echo "All instances ready. Run your benchmark with:"
echo ""
echo "  COMFYUI_ADDRS=${ADDRS_STR} python experiments/run_benchmark.py \\"
echo "      --model longcat --benchmark dpg-bench --parallel ${N_INSTANCES} ..."
echo ""
echo "Press Ctrl+C to stop all instances."
echo ""

wait
