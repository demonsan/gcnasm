#!/usr/bin/env bash
# Start the 4 ranks of fp8_a2a_sdma (non-MPI mode) on one node and collect their output.
#   ./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe --case dense --M 2048 --puts-per-peer 2 --half-first
# DEVICES  : comma list of the 4 GPUs (default 0,1,2,3). It is exported as HIP_VISIBLE_DEVICES to EVERY rank (CCO maps peer memory, so each process must see all 4 GPUs);
#            rank r then uses visible device r (--device is added automatically).
# WORKDIR  : directory for the uid file, the per-run sync directory and the logs (default: fresh mktemp dir; always fresh, stale files would be misread).
# TIMEOUT_S: bounded wait for the uid file / peers (default 120).
set -u
[ $# -ge 1 ] || { echo "usage: $0 <exe> [options...]" >&2; exit 2; }
EXE="$1"; shift
DEVICES="${DEVICES:-0,1,2,3}"
TIMEOUT_S="${TIMEOUT_S:-120}"
WORKDIR="${WORKDIR:-$(mktemp -d /tmp/fp8_a2a_sdma.XXXXXX)}"
mkdir -p "${WORKDIR}"
UID_FILE="${WORKDIR}/cco.uid"
rm -rf "${UID_FILE}" "${UID_FILE}.d"
export HIP_VISIBLE_DEVICES="${DEVICES}"
pids=()
for r in 0 1 2 3; do
  "${EXE}" --rank "${r}" --world 4 --uid-file "${UID_FILE}" --device "${r}" --sync-timeout-s "${TIMEOUT_S}" "$@" > "${WORKDIR}/rank${r}.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in 0 1 2 3; do wait "${pids[$i]}" || { echo "rank ${i} exited with status $?" >&2; rc=1; }; done
for r in 0 1 2 3; do echo "----- rank ${r} -----"; cat "${WORKDIR}/rank${r}.log"; done
echo "logs: ${WORKDIR}"
exit ${rc}