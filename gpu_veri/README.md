# Mesh-Rail A2A：8×A100 真机验证

本项目把论文 *Towards Extreme Token/$ Optimized Training System, A Wafer-Scale Exploration* 的三阶段 All-to-All 映射到单节点 8×A100：

```text
S1  Y-Allgather        0↔2, 1↔3, 4↔6, 5↔7   每向 D
S2  Inter-chip A2A     0↔4, 1↔5, 2↔6, 3↔7   每向 D/2
S3  X-Scatter          0↔1, 2↔3, 4↔5, 6↔7   每向 D/2
```

程序的目标是产生可以与 ASTRA-SIM 逐点比较的可复现实验数据，不是优化 NCCL 峰值性能。快层被限制到 40 GB/s，慢层被限制到 40/7.5 GB/s。限速采用同一 CUDA stream 上的 4 MiB 分块和 `%globaltimer` pacing kernel。

## 依赖

- 单节点 8×NVIDIA A100，GPU 间 P2P 可达（目标平台为 NVSwitch）
- CUDA Toolkit 11.8 或更新版本
- NCCL 2.17 或更新版本及开发头文件
- CMake 3.24+、支持 C++17 的主机编译器
- Python 3.9+（仅结果分析使用，无第三方包）
- `nvidia-smi`；可选 `numactl`

当前开发环境没有 NCCL 开发包且 GPU 被容器屏蔽，所以这里只能完成 CPU 静态验收；CUDA/NCCL 执行必须在目标服务器进行。

## 推荐的一键运行

先确认 `nvidia-smi topo -m` 中 8 卡映射符合预期，再运行：

```bash
cd gpu_veri
CPU_NODE=0 CLOCK_MHZ=1215 DEVICE_MAP=0,1,2,3,4,5,6,7 \
  ./scripts/run_a100.sh
```

脚本会依次：

1. 记录拓扑与 GPU 信息；
2. 尝试启用 persistence mode 并锁到 1215 MHz；
3. 构建并运行 CPU 静态检查；
4. 标定 `B_raw`、`L_eff` 和每 chunk 的 `O_ns`，在 D=2 MiB/256 MiB 诊断单一带宽模型；
5. 扫描 5 个 D 和 3 个独立阶段；
6. 对 S1 做 1/2/4 pair 并发扫描；
7. 为五个 D 分别生成 ASTRA-SIM 网络配置和验收报告，最后恢复时钟设置。

锁频命令通常需要管理员权限。可先由管理员执行：

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1215,1215
```

若脚本无法锁频，它会继续运行并将每次迭代的两端频率写入 `*.samples.csv`。报告会给出频率分布及按众数筛样本的数量。正式 rebuttal 数据优先使用成功锁频的运行。

若时钟已由宿主机管理员锁定，容器内使用 `CLOCK_PRELOCKED=1 CLOCK_MHZ=1215 ./scripts/run_a100.sh`。脚本会严格核对 8 卡当前频率，但不会在退出时恢复宿主机时钟。

无宿主机锁频权限时，使用 `SKIP_CLOCK_LOCK=1 ITERS=300 WARMUP=50 ./scripts/run_a100.sh`。程序不会尝试修改任何 GPU 状态；在线标定先按每个配置两端 GPU 的频率众数抑制异常样本，最终分析再按整个 sweep 中每张物理 GPU 的全局频率众数重新计算统计量，写入 `results.filtered.csv`。逐迭代原始数据仍完整保留在 `*.samples.csv`，全局筛选后任一点少于 100 个样本会使严格验收失败。

只有两张空闲卡时可直接运行：

```bash
SKIP_CLOCK_LOCK=1 ITERS=300 WARMUP=50 FAST_BW=12 \
DEVICE_MAP=6,7 RUN_CONCURRENCY=0 RESULT_DIR="$PWD/results/a100_gpu67_bw12" \
  ./scripts/run_a100.sh
```

`FAST_BW` 默认为 40 GB/s。`RUN_CONCURRENCY=auto` 在 `DEVICE_MAP` 有 8 项时运行并发扫描，有 2 项时自动跳过；也可显式设为 `0` 或 `1`。

`CPU_NODE` 应根据 `nvidia-smi topo -m` 的 CPU affinity 选择；不设置时不启用 NUMA 绑定。8 项 `DEVICE_MAP` 的第 i 项是论文逻辑 rank i 使用的 CUDA device ID。

`calibrate`、`stage` 和 `sweep` 同时只驱动一对卡，也可用紧凑映射 `--device-map 0,1`；程序会让三个代表 pair 顺序复用这两张物理卡，同时保留论文中的逻辑 pair 标签。`concurrency` 必须给出 8 个互不重复的设备 ID，因为 4-pair 点确实需要 8 卡。

## 手工构建与运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

先标定，顺序不能交换：

```bash
./build/a2a_bench \
  --mode calibrate --size 32M --fast-bw 40 --ratio 7.5 --chunk 4M \
  --iters 100 --warmup 20 --expected-clock 1215 --strict-clock \
  --device-map 0,1,2,3,4,5,6,7 --verify \
  --calibration results/calibration.json --out results/calibration.csv
```

完整扫描：

```bash
./build/a2a_bench \
  --mode sweep --fast-bw 40 --ratio 7.5 --chunk 4M \
  --iters 100 --warmup 20 --expected-clock 1215 --strict-clock \
  --device-map 0,1,2,3,4,5,6,7 --verify \
  --calibration results/calibration.json --out results/results.csv
```

单阶段调试和并发度扫描：

```bash
./build/a2a_bench --mode stage --stage s2 --size 32M \
  --calibration results/calibration.json --out results/s2.csv --verify

./build/a2a_bench --mode concurrency --size 32M \
  --calibration results/calibration.json --out results/concurrency.csv --verify
```

手工生成报告：

```bash
python3 scripts/analyze_results.py \
  --csv results/results.csv \
  --samples results/results.samples.csv \
  --concurrency results/concurrency.csv \
  --calibration results/calibration.json \
  --out results/FINDINGS.md \
  --astra-dir results/astra_sim_profiles --strict
```

## 输出口径

`results.csv` 使用计划书规定的汇总 schema，每个 D 有 S1/S2/S3 三行。`t_median_us` 是一对 GPU 两端 event 时间的较大值的中位数；`achieved_bw_gbps = per_link_bytes / t_median`。GB/s 使用十进制定义，输入的 `M` 使用 MiB。

对应的 `results.samples.csv` 保存每次迭代的两端 event 时间、物理 GPU ID 和两端 SM 频率，供无 root 场景筛除频率漂移样本。并发模式在主表的 `mode` 字段中使用 `concurrency-1/2/4`，避免修改既定 CSV schema。

`calibration.json` 保存 pacing 标定和 1-byte 操作延迟：

- `latency_us`：1-byte 成对操作的有效延迟；
- `O_ns`：软件 pacing 的每 chunk 固定开销，不是 ASTRA-SIM 网络参数；
- `sm_clock_locked*` 和 `pace_timer`：复现性元数据。

分析阶段额外生成 `astra_sim_profiles/`：

- `network_D_2MiB.yml`、`network_D_8MiB.yml`、`network_D_32MiB.yml`、`network_D_64MiB.yml`、`network_D_256MiB.yml`：可直接用于对应 D 的 ASTRA-SIM analytical network 配置；
- `profiles.json`：保留每个阶段的实测时间、实测有效带宽、ASTRA-SIM 序列化带宽、延迟来源和维度映射。

生成 GPU profile 后，可按 [`astra_sim/README.md`](astra_sim/README.md) 运行
2×(2×2) ASTRA-SIM 对照实验。脚本会自动探测 `comm_size` 语义、运行两个
analytical 后端，并生成逐点误差表。

ASTRA-SIM 配置的维度顺序是 `[X,Y,W]=[S3,S1,S2]`。实测有效带宽已经包含启动延迟，不能与非零 latency 直接同时填写；生成器使用 `B_astra=bytes/(t_median-L_eff)` 反推序列化带宽，使 `L_eff+bytes/B_astra` 恢复该点的实测 median。

若修改时钟、chunk、目标带宽、NCCL/CUDA 版本或拓扑，必须重新标定。阶段/sweep 会拒绝加载 chunk 或目标带宽不一致的标定文件。

## 内置验收与故障处理

- 程序启动时报告 CUDA/NCCL 版本、P2P 矩阵和 `nvidia-smi topo -m`；NCCL <2.17 或计算能力 <sm_70 会退出。
- `%globaltimer` 自检使用 10 ms kernel，median 误差超过 1% 会退出。
- `B_raw < 3×B_target` 或闭环 10 轮不收敛会使标定失败。
- D=2/256 MiB 未达到 3% 会记录建模警告，但不再阻止 sweep；五个消息点分别使用其实测 ASTRA-SIM profile。
- 任一 chunk 算得 `t_pace≤0` 会告警并令 `pace_clamped=1`；严格分析会拒绝该结果。
- `--verify` 在计时外逐字节校验 AllGather 的 chunk-major 输出及 P2P 对端数据。
- 代码只接受三类合法 pair；CPU 静态检查同时断言四条对角线链路通信次数为零。

若 2 MiB 和 256 MiB 无法同时达到 3%，说明固定开销不是纯加性。报告会保留该警告并改用逐消息实测配置；不要把这种逐点回放描述成跨消息大小预测。建议在同一整套实验中固定 NCCL 环境变量，并把最终值写入实验日志。

## 结论边界

逐消息 profile 模式验证的是三阶段映射、通信量以及 ASTRA-SIM 在给定实测参数下的逐点复现能力，不再验证单一带宽模型的跨消息大小预测。它不能验证 6×6 mesh 的中心热点、多跳路由或 16 wafer torus 的调度，也不能把三阶段单 pair 时间之和解释为完整 8 卡算法的真实 wall time。详细声明由分析脚本写入 `FINDINGS.md`。
