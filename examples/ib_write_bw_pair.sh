#!/usr/bin/env bash
# Hardware-only site template; never used by synthetic mode.
# It validates transport inputs and captures both endpoint lifecycles, but one invocation covers only one configured rail.
set -euo pipefail
: "${PEER_HOST:?set PEER_HOST to the paired InfiniBand hostname/address}"
[[ "${PEER_HOST}" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,252}$ ]] || { echo "unsafe PEER_HOST" >&2; exit 64; }
[[ -z "${IB_DEVICE:-}" || "${IB_DEVICE}" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "unsafe IB_DEVICE" >&2; exit 64; }
[[ -z "${IB_PORT:-}" || "${IB_PORT}" =~ ^[1-9][0-9]*$ ]] || { echo "unsafe IB_PORT" >&2; exit 64; }
base_port="${CLUSTERCHECK_IB_BASE_PORT:-18515}"
job_component="${SLURM_JOB_ID:-0}"
[[ "${base_port}" =~ ^[1-9][0-9]*$ && "${job_component}" =~ ^[0-9]+$ ]] || { echo "invalid port/job value" >&2; exit 64; }
port="${CLUSTERCHECK_IB_PORT:-$((base_port + job_component % 30000))}"
(( port >= 1024 && port <= 65535 )) || { echo "IB port out of range" >&2; exit 64; }
device_args=()
[[ -n "${IB_DEVICE:-}" ]] && device_args+=(--ib-dev="${IB_DEVICE}")
[[ -n "${IB_PORT:-}" ]] && device_args+=(--ib-port="${IB_PORT}")
server_log="$(mktemp)"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then kill -- "-${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; fi
  rm -f "${server_log}"
}
trap cleanup EXIT INT TERM
remote=(ib_write_bw --report_gbits --port="${port}" "${device_args[@]}")
setsid ssh -o BatchMode=yes -o ConnectTimeout=10 -- "${PEER_HOST}" timeout 240 "${remote[@]}" >"${server_log}" 2>&1 &
server_pid=$!
sleep "${CLUSTERCHECK_IB_SERVER_DELAY:-2}"
if ! ib_write_bw --report_gbits --port="${port}" "${device_args[@]}" "${PEER_HOST}"; then
  cat "${server_log}" >&2
  exit 1
fi
if ! wait "${server_pid}"; then
  cat "${server_log}" >&2
  exit 1
fi
server_pid=""
cat "${server_log}" >&2
rm -f "${server_log}"
trap - EXIT INT TERM
