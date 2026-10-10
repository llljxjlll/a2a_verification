#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
result_dir="${RESULT_DIR:-${project_dir}/results}"
clock_mhz="${CLOCK_MHZ:-1215}"
clock_prelocked="${CLOCK_PRELOCKED:-0}"
skip_clock_lock="${SKIP_CLOCK_LOCK:-0}"
iters="${ITERS:-100}"
warmup="${WARMUP:-20}"
device_map="${DEVICE_MAP:-0,1,2,3,4,5,6,7}"
fast_bw="${FAST_BW:-40}"
run_concurrency="${RUN_CONCURRENCY:-auto}"
cpu_node="${CPU_NODE:-}"
lock_succeeded=0

IFS=',' read -r -a device_ids <<< "${device_map}"
if [[ "${run_concurrency}" == auto ]]; then
  if [[ "${#device_ids[@]}" == 8 ]]; then
    run_concurrency=1
  else
    run_concurrency=0
  fi
fi
if [[ "${run_concurrency}" != 0 && "${run_concurrency}" != 1 ]]; then
  echo "ERROR: RUN_CONCURRENCY must be auto, 0, or 1." >&2
  exit 1
fi
if [[ "${run_concurrency}" == 1 && "${#device_ids[@]}" != 8 ]]; then
  echo "ERROR: RUN_CONCURRENCY=1 requires an 8-entry DEVICE_MAP." >&2
  exit 1
fi

mkdir -p "${result_dir}"
exec > >(tee "${result_dir}/run.log") 2>&1

echo "== Run metadata =="
date --iso-8601=seconds
hostname
uname -a
echo "DEVICE_MAP=${device_map} CPU_NODE=${cpu_node:-unset} FAST_BW=${fast_bw} RUN_CONCURRENCY=${run_concurrency} CLOCK_MHZ=${clock_mhz} CLOCK_PRELOCKED=${clock_prelocked} SKIP_CLOCK_LOCK=${skip_clock_lock} ITERS=${iters} WARMUP=${warmup}"
env | grep '^NCCL_' | sort || true

restore_clock() {
  if [[ "${lock_succeeded}" == 1 ]]; then
    nvidia-smi -rgc >/dev/null || true
  fi
}
trap restore_clock EXIT

echo "== GPU topology =="
nvidia-smi topo -m
echo "== GPU inventory =="
nvidia-smi --query-gpu=index,name,memory.total,clocks.sm,persistence_mode --format=csv

if [[ "${clock_prelocked}" == 1 && "${skip_clock_lock}" == 1 ]]; then
  echo "ERROR: CLOCK_PRELOCKED and SKIP_CLOCK_LOCK cannot both be 1." >&2
  exit 1
elif [[ "${clock_prelocked}" == 1 ]]; then
  if ! nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | \
       awk -v expected="${clock_mhz}" '$1 != expected { bad=1 } END { exit bad }'; then
    echo "ERROR: CLOCK_PRELOCKED=1, but at least one GPU is not at ${clock_mhz} MHz." >&2
    exit 1
  fi
  echo "Using host-prelocked GPU clocks; this script will not reset them."
  clock_args=(--expected-clock "${clock_mhz}" --strict-clock)
elif [[ "${skip_clock_lock}" == 1 ]]; then
  echo "Running without clock lock; frequency-mode filtering is enabled."
  clock_args=()
elif nvidia-smi -pm 1 && nvidia-smi -lgc "${clock_mhz},${clock_mhz}"; then
  lock_succeeded=1
  clock_args=(--expected-clock "${clock_mhz}" --strict-clock)
else
  echo "WARNING: clock lock failed; retaining per-sample clocks for mode filtering." >&2
  clock_args=()
fi

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j
ctest --test-dir "${build_dir}" --output-on-failure

runner=()
if [[ -n "${cpu_node}" ]]; then
  runner=(numactl "--cpunodebind=${cpu_node}" "--membind=${cpu_node}")
fi

common=(
  --fast-bw "${fast_bw}"
  --ratio 7.5
  --chunk 4M
  --iters "${iters}"
  --warmup "${warmup}"
  --device-map "${device_map}"
  --calibration "${result_dir}/calibration.json"
  --verify
  "${clock_args[@]}"
)

"${runner[@]}" "${build_dir}/a2a_bench" \
  --mode calibrate --size 32M --out "${result_dir}/calibration.csv" "${common[@]}"

"${runner[@]}" "${build_dir}/a2a_bench" \
  --mode sweep --out "${result_dir}/results.csv" "${common[@]}"

analysis_concurrency=()
if [[ "${run_concurrency}" == 1 ]]; then
  "${runner[@]}" "${build_dir}/a2a_bench" \
    --mode concurrency --size 32M --out "${result_dir}/concurrency.csv" "${common[@]}"
  analysis_concurrency=(--concurrency "${result_dir}/concurrency.csv")
else
  echo "Skipping concurrency scan; it requires 8 distinct GPUs."
fi

python3 "${project_dir}/scripts/analyze_results.py" \
  --csv "${result_dir}/results.csv" \
  --samples "${result_dir}/results.samples.csv" \
  "${analysis_concurrency[@]}" \
  --calibration "${result_dir}/calibration.json" \
  --out "${result_dir}/FINDINGS.md" \
  --astra-dir "${result_dir}/astra_sim_profiles" \
  --strict

echo "Results are in ${result_dir}"
