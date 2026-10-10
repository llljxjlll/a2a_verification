# A2A Verification

This repository contains the 8×A100 validation project for the three-stage
All-to-All communication model described in the accompanying HPCA 2027 work.

## Contents

- `gpu_veri/`: CUDA/NCCL benchmark, experiment runner, analysis scripts, and
  the validation plan.
- `astra-sim/`: the upstream ASTRA-SIM revision used as the simulator
  reference, tracked as a Git submodule.

The anonymous submission PDF is intentionally excluded because this repository
is public.

## Clone

```bash
git clone --recurse-submodules https://github.com/llljxjlll/a2a_verification.git
cd a2a_verification
```

See [`gpu_veri/README.md`](gpu_veri/README.md) for requirements and instructions
for running the experiment on an 8×A100 server.
