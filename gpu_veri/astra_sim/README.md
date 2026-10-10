# ASTRA-SIM：2×(2×2) Mesh-Rail 对照实验

这里的脚本读取 GPU 实测生成的五组 `astra_sim_profiles`，构建 8-NPU、三维
`[X,Y,W]=[2,2,2]` 系统，并逐点运行论文中的三个串行阶段：

```text
S1: ALL_GATHER     on Y, per-link bytes = D
S2: ALL_TO_ALL     on W, per-link bytes = D/2
S3: REDUCE_SCATTER on X, per-link bytes = D/2
```

网络维度均使用 `FullyConnected`。当维度大小为 2 时，它表示一条双向链路，避免
`Ring` 可能引入的双链路口径。脚本不会假定 Chakra `comm_size` 的含义；每次运行
先在 100 GB/s、零延迟网络上执行 L0 探针，再反解三个 collective 各自的倍率。

锁定版本的 analytical backend 源码把配置中的 1 GB 定义为 `2^30 B`，而 GPU
结果使用十进制 `10^9 B`。运行器会自动把实测带宽乘以 `10^9/2^30` 后写入网络
配置，避免仿真结果静默快约 7.37%。

`endpoint-delay` 使用 1 ns 而不是数学上的 0：锁定的 analytical event queue 要求
事件时间严格递增，0 会触发断言失败。该 1 ns 仅用于保证事件推进，在最小的 2 MiB
实测（216 µs 以上）中低于 0.0005%，不会改变对照结论。

该版本的 `Congestion_Aware` executable 明确只支持一维拓扑，无法载入三维
2×2×2 网络。完整 L0–L4 因而使用支持多维拓扑的 `Congestion_Unaware`，报告会
显式记录这一限制，不会把一维结果冒充为三维拥塞仿真。

## 依赖准备

从项目根目录确认所有子模块（包括 ASTRA-SIM 的嵌套子模块）完整：

```bash
git submodule update --init --recursive
```

ASTRA-SIM analytical 后端构建需要 CMake、C++17、protobuf 编译器/开发库和 Python
protobuf。Ubuntu 容器缺少这些包时可安装：

```bash
apt-get update
apt-get install -y build-essential cmake protobuf-compiler libprotobuf-dev python3-protobuf
```

无 sudo 的 Conda 环境可使用：

```bash
conda install -y -n base -c conda-forge libprotobuf protobuf
```

`run_all.sh` 会自动使用 Conda protobuf 的 CMake Config target，把新版 protobuf
所需的 Abseil 传递依赖一并链接。

构建过程中 CMake 还会获取 `yaml-cpp`，因此首次构建需要访问 GitHub。仿真是 CPU
程序，不使用 GPU，也不需要锁频或设置 `CUDA_VISIBLE_DEVICES`。

## 运行

假设 GPU 实测结果位于 `gpu_veri/results/a100_gpu67_bw12_v2`：

```bash
cd /root/work/a2a_veri/a2a_verification
GPU_RESULT_DIR="$PWD/gpu_veri/results/a100_gpu67_bw12_v2" \
SIM_RESULT_DIR="$PWD/gpu_veri/results/a100_gpu67_bw12_v2/astra_sim_run" \
  ./gpu_veri/astra_sim/run_all.sh
```

首次运行会编译支持多维网络的 congestion-unaware analytical 后端。以后复跑可跳过构建：

```bash
BUILD_ASTRA=0 \
GPU_RESULT_DIR="$PWD/gpu_veri/results/a100_gpu67_bw12_v2" \
  ./gpu_veri/astra_sim/run_all.sh
```

如 ASTRA-SIM 不在仓库默认的 `astra-sim/`，用 `ASTRA_SIM_DIR` 指定路径。

## 输出

- `astra_results.csv`：含 `D, stage, dim, comm_size, per_link_bytes, backend,
  t_us`，并附解析公式、GPU median 及两类误差；
- `l2_results.csv`：同一 All-to-All 在 Y-fast 与 W-slow 上的耗时和比值；
- `FINDINGS_sim.md`：L0–L4 逐级验收和 analytical 后端适用性说明；
- `comm_size_semantics.json`：本版本 ASTRA-SIM 实测得到的三种 collective 倍率；
- `workloads/`：五个消息点的单阶段、串联和 L2 维度探针 ET；
- `networks/`：由 `profiles.json` 重新物化的五组 `FullyConnected` 网络配置；
- `logs/`：每次 ASTRA-SIM 原始标准输出，便于追溯。

每个网络 profile 来自同一消息大小的 GPU 实测数据，因此这项实验是逐点复现，不能
表述为一个参数集对未知消息大小的预测。完整建模边界见生成的 `FINDINGS_sim.md`。
