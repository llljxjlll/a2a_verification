#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_veri_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${gpu_veri_dir}/.." && pwd)"
astra_sim_dir="${ASTRA_SIM_DIR:-${repo_dir}/astra-sim}"
gpu_result_dir="${GPU_RESULT_DIR:-${gpu_veri_dir}/results}"
sim_result_dir="${SIM_RESULT_DIR:-${gpu_result_dir}/astra_sim_run}"
build_astra="${BUILD_ASTRA:-1}"

profiles="${gpu_result_dir}/astra_sim_profiles/profiles.json"
gpu_csv="${gpu_result_dir}/results.filtered.csv"
if [[ ! -f "${gpu_csv}" ]]; then
  gpu_csv="${gpu_result_dir}/results.csv"
fi

if [[ ! -f "${profiles}" ]]; then
  echo "ERROR: missing ${profiles}" >&2
  echo "Run the GPU sweep/analyzer first, or set GPU_RESULT_DIR to that run directory." >&2
  exit 2
fi
if [[ ! -f "${gpu_csv}" ]]; then
  echo "ERROR: missing GPU result CSV in ${gpu_result_dir}" >&2
  exit 2
fi
if [[ ! -f "${astra_sim_dir}/build/astra_analytical/build.sh" ]]; then
  echo "ERROR: ASTRA-SIM submodule is unavailable at ${astra_sim_dir}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 2
fi
if [[ ! -f "${astra_sim_dir}/extern/graph_frontend/chakra/schema/protobuf/et_def.proto" ]]; then
  echo "ERROR: ASTRA-SIM nested submodules are not initialized." >&2
  echo "Run from ${repo_dir}: git submodule update --init --recursive" >&2
  exit 2
fi
if ! python3 -c 'import google.protobuf' >/dev/null 2>&1; then
  echo "ERROR: Python protobuf is not installed (Ubuntu package: python3-protobuf)." >&2
  exit 2
fi

if [[ "${build_astra}" == 1 ]]; then
  for command in cmake protoc; do
    if ! command -v "${command}" >/dev/null 2>&1; then
      echo "ERROR: required build command '${command}' is not installed." >&2
      exit 2
    fi
  done
  echo "== Building both ASTRA-SIM analytical backends =="
  "${astra_sim_dir}/build/astra_analytical/build.sh" -t all
elif [[ "${build_astra}" != 0 ]]; then
  echo "ERROR: BUILD_ASTRA must be 0 or 1." >&2
  exit 2
fi

echo "== Running L0-L4 on 2x(2x2) Mesh-Rail =="
mkdir -p "${sim_result_dir}"
python3 "${script_dir}/run_astra_sim.py" \
  --astra-sim-dir "${astra_sim_dir}" \
  --profiles "${profiles}" \
  --gpu-csv "${gpu_csv}" \
  --output-dir "${sim_result_dir}" \
  --system "${script_dir}/system.json" \
  --probe-network "${script_dir}/network_probe.yml" \
  --memory "${script_dir}/no_memory_expansion.json" \
  --strict

echo "Results: ${sim_result_dir}/astra_results.csv"
echo "Report:  ${sim_result_dir}/FINDINGS_sim.md"
