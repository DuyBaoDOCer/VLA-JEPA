#!/bin/bash
# Sequential single-GPU LIBERO eval across all 4 task suites (TIP-010 N1).
#
# eval_libero.sh assumes >=5 GPUs: it backgrounds all 4 suites at once on
# --cuda 1..4, each against its own server_policy.py instance. We have
# exactly 1 GPU (index 0), so each suite runs to completion -- server up,
# eval done, server killed, port closed -- before the next one starts.
# README.md's Notes section explicitly allows rewriting the
# parallelization logic; this does not touch eval_libero.py or
# server_policy.py themselves.
#
# Required environment variables (set by the caller, e.g. the notebook):
#   CKPT_PATH      local path to the .pt checkpoint, in the
#                  <RUN_DIR>/checkpoints/<name>.pt layout
#                  share_tools.py:read_mode_config requires
#   POLICY_PYTHON  env-policy's python interpreter (runs server_policy.py)
#   SIM_PYTHON     env-sim's python interpreter (runs eval_libero.py)
#   RESULTS_DIR    base directory for per-suite logs and videos
# Assumed already exported by the caller, matching eval_libero.sh's own
# pattern: LIBERO_HOME, LIBERO_CONFIG_PATH, PYTHONPATH, MUJOCO_GL.
#
# Per suite, writes under "${RESULTS_DIR}/${suite}/":
#   server.log     server_policy.py's stdout+stderr
#   eval.log       eval_libero.py's stdout+stderr (success-rate lines live here)
#   wall_s.txt     integer seconds eval_libero.py itself took for this suite
#   exit_code.txt  eval_libero.py's exit code, or "server_start_timeout"
#                  if server_policy.py never opened its port
# No single-line summary file: the notebook's own Python-side parser reads
# eval.log directly and reconstructs any rich-wrapped log lines itself
# (TIP-009d's lesson, C33c) rather than duplicating that logic in bash.

set -uo pipefail

: "${CKPT_PATH:?CKPT_PATH must be set to the local .pt checkpoint path}"
: "${POLICY_PYTHON:?POLICY_PYTHON must be set to the env-policy python interpreter}"
: "${SIM_PYTHON:?SIM_PYTHON must be set to the env-sim python interpreter}"
: "${RESULTS_DIR:?RESULTS_DIR must be set to a base results directory}"

HOST="127.0.0.1"
BASE_PORT=15083
NUM_TRIALS_PER_TASK=1
WITH_STATE="true"
SUITES=("libero_10" "libero_goal" "libero_object" "libero_spatial")

PORT_OPEN_TIMEOUT_S=60
PORT_CLOSE_TIMEOUT_S=20

# Plain bash TCP port probes via the /dev/tcp pseudo-device (a bash
# builtin, no netcat/nc dependency needed) -- polls instead of a blind
# sleep, so this fails fast and loudly if the server never comes up
# instead of racing eval_libero.py against an unready port.
wait_for_port_open() {
    local host="$1" port="$2" timeout_s="$3"
    local waited=0
    while true; do
        if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
            exec 3>&- 3<&- 2>/dev/null || true
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$timeout_s" ]; then
            return 1
        fi
    done
}

wait_for_port_closed() {
    local host="$1" port="$2" timeout_s="$3"
    local waited=0
    while true; do
        if ! (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
            return 0
        fi
        exec 3>&- 3<&- 2>/dev/null || true
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$timeout_s" ]; then
            return 1
        fi
    done
}

mkdir -p "${RESULTS_DIR}"

index=0
for task_suite_name in "${SUITES[@]}"; do
    index=$((index + 1))
    port=$((BASE_PORT + index))
    video_out_path="${RESULTS_DIR}/${task_suite_name}"
    mkdir -p "${video_out_path}"

    echo "=== [${task_suite_name}] starting server_policy.py on port ${port} ==="
    "${POLICY_PYTHON}" ./deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        > "${video_out_path}/server.log" 2>&1 &
    server_pid=$!

    if ! wait_for_port_open "${HOST}" "${port}" "${PORT_OPEN_TIMEOUT_S}"; then
        echo "=== [${task_suite_name}] FAILED: server_policy.py did not open port ${port} within ${PORT_OPEN_TIMEOUT_S}s -- see ${video_out_path}/server.log ==="
        echo "server_start_timeout" > "${video_out_path}/exit_code.txt"
        echo 0 > "${video_out_path}/wall_s.txt"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        continue
    fi
    echo "=== [${task_suite_name}] server up (pid ${server_pid}), running eval_libero.py ==="

    suite_start=$(date +%s)
    "${SIM_PYTHON}" ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${CKPT_PATH}" \
        --args.host "${HOST}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.video-out-path "${video_out_path}" \
        --args.with_state "${WITH_STATE}" \
        > "${video_out_path}/eval.log" 2>&1
    eval_exit=$?
    suite_end=$(date +%s)
    echo "${eval_exit}" > "${video_out_path}/exit_code.txt"
    echo $((suite_end - suite_start)) > "${video_out_path}/wall_s.txt"

    if [ "${eval_exit}" -ne 0 ]; then
        echo "=== [${task_suite_name}] FAILED: eval_libero.py exited ${eval_exit} -- see ${video_out_path}/eval.log ==="
    else
        echo "=== [${task_suite_name}] OK: wall_s=$((suite_end - suite_start)) -- see ${video_out_path}/eval.log for success rate ==="
    fi

    echo "=== [${task_suite_name}] stopping server_policy.py (pid ${server_pid}) ==="
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    if ! wait_for_port_closed "${HOST}" "${port}" "${PORT_CLOSE_TIMEOUT_S}"; then
        echo "=== [${task_suite_name}] WARNING: port ${port} did not close within ${PORT_CLOSE_TIMEOUT_S}s, continuing anyway ==="
    fi
done

echo "=== eval_libero_1gpu.sh done, per-suite logs under ${RESULTS_DIR}/<suite>/ ==="
