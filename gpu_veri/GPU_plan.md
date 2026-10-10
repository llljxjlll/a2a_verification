---
tags: [科研, 晶圆级通信, HPCA2027, rebuttal, 开发计划]
status: draft
created: 2026-10-10
updated: 2026-10-10
---

# GPU 模拟 A2A 算法 — 开发计划书

> **本文档是给实现 agent 看的完整规格说明，应可独立阅读，不依赖其他上下文。**
> 相关背景见 [[rebuttal_HPCA2027]]。

---

## 0. 任务目标

在**单节点 8×A100** 上实现一个基准测试程序，用于：

1. **提取链路参数**（有效带宽 `B_eff`、有效延迟 `L_eff`），用于配置 ASTRA-SIM 仿真器；
2. **实测** Mesh-Rail 论文提出的**三阶段 All-to-All 算法**各阶段耗时；
3. 产出可与 ASTRA-SIM 仿真结果逐点对照的 CSV 数据。

最终用途：论文 rebuttal，回应审稿人"性能论断完全基于分析模型，缺少真机测试平台"的质疑。

### ⚠️ 实现者必读：这不是性能优化任务

本程序的价值**不在于跑得快**，而在于**通信行为可被精确预测**。

- **请严格按 §4 规定的通信量实现**，不要为了"更高效"而合并、省略或改变任何一个阶段的传输量。
- **请勿**调整算法结构、通信模式或数据布局去追求更好的数字。
- 实测结果将与 ASTRA-SIM 仿真逐点对照，**任何偏离规格的"优化"都会让对照失效**。

唯一的优化目标是：**测量准确、可复现、与解析预测一致**。

---

## 1. 背景知识

### 1.1 论文与系统架构

论文 *"Towards Extreme Token/$ Optimized Training System, A Wafer-Scale Exploration"*（Mesh-Rail，HPCA 2027 投稿）提出一种晶圆级 LLM 训练系统，含三级网络：

| 层级 | 拓扑 | 带宽 / 延迟 |
|---|---|---|
| 片内 WSC（晶圆级芯片） | 6×6 2D **mesh** | 3.43 TB/s；2 ns |
| Scale-up | 16 个 WSC 组成 2D **torus** | 460 GB/s；100 ns |
| Scale-out | X/Y 2D rail-only 交换网络 | 400 Gb/s；500 ns |

**关键比值**：片内 : scale-up ≈ **7.5 : 1**。本实验需复现这个带宽分层。

### 1.2 All-to-All 的来源

A2A 来自 MoE 模型的 **专家并行（EP）dispatch / combine**。它跨越片内 mesh 和片间 torus 两级网络。

### 1.3 论文方法：三阶段 A2A

传统做法是分层 A2A（片内 A2A → 片间 A2A），但片内那一跳是在 2D mesh 上做任意点对点全交换，流量会压在中心链路产生热点与拥塞。

论文把 A2A 重写为三个阶段：

```
Y-Allgather  →  Inter-chip A2A  →  X-Scatter
   (片内)          (片间 torus)       (片内)
```

1. **Y-Allgather**：WSC 内部沿 **Y 维**（同列的 die）做 all-gather。
2. **Inter-chip A2A**：跨 WSC，**只在下标相同的 die 之间**（rail 式配对）做 A2A。
3. **X-Scatter**：WSC 内部沿 **X 维** 把数据分发到最终目的地。

**核心思想**：A2A 本质是数据块的置换。先沿 Y 聚合后，片间阶段不再需要任意的片内路由；最后沿 X 散射到目的地。效果是所有片内流量被约束在**一维**，天然无交叉、无热点、无需多跳转发。

**代价**：Y-Allgather 引入冗余数据传输，通信量不再最优。论文的论点是"用多搬一点数据换完全无拥塞"。

论文声称（仿真结果，6×6 mesh × 16 wafer = 576 die 规模）：带宽利用率 1.79×，延迟 −44%。

---

## 2. 系统映射：8 GPU → 2×(2×2)

8 块 GPU 映射为 **2 个 wafer × (2×2 mesh)**。维度序按 ASTRA-SIM 约定**内层在前**：`[X, Y, W]`。

```
rank = x + 2*y + 4*w
```

| GPU | x | y | w | 位置 |
|:--:|:--:|:--:|:--:|---|
| 0 | 0 | 0 | 0 | Wafer0 (0,0) |
| 1 | 1 | 0 | 0 | Wafer0 (1,0) |
| 2 | 0 | 1 | 0 | Wafer0 (0,1) |
| 3 | 1 | 1 | 0 | Wafer0 (1,1) |
| 4 | 0 | 0 | 1 | Wafer1 (0,0) |
| 5 | 1 | 0 | 1 | Wafer1 (1,0) |
| 6 | 0 | 1 | 1 | Wafer1 (0,1) |
| 7 | 1 | 1 | 1 | Wafer1 (1,1) |

### 链路清单

**快层（片内 mesh）**

| wafer | X 链路 | Y 链路 |
|---|---|---|
| W0 | 0–1, 2–3 | 0–2, 1–3 |
| W1 | 4–5, 6–7 | 4–6, 5–7 |

**慢层（scale-up rail）**：0–4, 1–5, 2–6, 3–7

**不存在的链路**：对角线 0–3、1–2、4–7、5–6。程序中**不得**在这些 GPU 之间直接通信。

> 三阶段算法的所有通信都落在合法链路上，**全程单跳，不需要任何转发**。

---

## 3. 实现范围：单对测试

2×(2×2) 下三阶段算法的每个阶段都是 **4 组独立的成对交换**，且每个 GPU 在每个阶段**只参与 1 对、只驱动 1 条链路**：

| 阶段 | 4 组配对 | 用到的链路 |
|---|---|---|
| S1 Y-Allgather | 0↔2, 1↔3, 4↔6, 5↔7 | 仅 Y 链路 |
| S2 Inter-chip A2A | 0↔4, 1↔5, 2↔6, 3↔7 | 仅 rail |
| S3 X-Scatter | 0↔1, 2↔3, 4↔5, 6↔7 | 仅 X 链路 |

四组结构全同、无共享资源，因此**测 1 组即可代表该阶段**。

> 结果：主体测量只需 **2 块 GPU**。仅 §6.3 的并发度扫描需要用到 4 / 8 块。

---

## 4. 各阶段规格

记 `D = data_size` = 每卡持有的 A2A 总 buffer（字节）。全系统 N=8 rank，故每目的地 block = D/8。

### 4.0 NCCL API 注意事项

**NCCL 没有 `ncclAllToAll`，也没有 `ncclScatter`。** 可用的 collective 只有：
`ncclAllReduce` / `ncclBroadcast` / `ncclReduce` / `ncclAllGather` / `ncclReduceScatter`，以及点对点 `ncclSend` / `ncclRecv`。

因此 A2A 与 Scatter 均须用 `ncclSend`+`ncclRecv` 包在 `ncclGroupStart()`/`ncclGroupEnd()` 中实现。

**数据类型**：使用 `ncclChar`（1 字节），使 `count == bytes`，避免单位换算错误。

### 4.1 S1 — Y-Allgather（快层）

- **代表对**：GPU **0 ↔ 2**
- **API**：`ncclAllGather`
- **sendbuff**：D 字节；**recvbuff**：**2D** 字节
- **count**：`D`
- **每向传输量**：**D**

```c
ncclAllGather(sendbuff, recvbuff, D, ncclChar, comm_Y, stream);
```

推导：n=2 的 allgather，每 rank 贡献整个 buffer D 给唯一的对端。

### 4.2 S2 — Inter-chip A2A（慢层）

- **代表对**：GPU **0 ↔ 4**
- **API**：`ncclSend` + `ncclRecv`（group 内）
- **count**：`D/2`
- **每向传输量**：**D/2**

```c
ncclGroupStart();
ncclSend(sendbuff, D/2, ncclChar, peer, comm_W, stream);
ncclRecv(recvbuff, D/2, ncclChar, peer, comm_W, stream);
ncclGroupEnd();
```

推导：allgather 后持 2D，负责其中 `y'==自己的 y` 的一半即 D；这 D 中去对面 wafer 的又是一半 → D/2。

### 4.3 S3 — X-Scatter（快层）

- **代表对**：GPU **0 ↔ 1**
- **API**：`ncclSend` + `ncclRecv`（group 内），**与 S2 结构完全相同**
- **count**：`D/2`
- **每向传输量**：**D/2**

推导：持 D 待分发给 m=2 个 X 位置，一半留本地一半给邻居 → D/2。

> **注意**：虽名为 "Scatter"，但所有 rank 同时各自分发，合起来是沿 X 维的 all-to-all 模式。m=2 时退化为对称成对交换，**不是**单 root 的 scatter。

> **实现提示**：S2 与 S3 可共用同一个函数，仅切换 communicator 与限速参数。

### 4.4 传输量汇总

| 阶段 | 代表对 | 层 | 每向字节 | NCCL API |
|---|---|---|---|---|
| S1 Y-Allgather | 0↔2 | 快 | `D` | `ncclAllGather`，count=D |
| S2 Inter-chip A2A | 0↔4 | 慢 | `D/2` | Send+Recv，count=D/2 |
| S3 X-Scatter | 0↔1 | 快 | `D/2` | Send+Recv，count=D/2 |

---

## 5. 带宽限制

### 5.1 目标值

| 层 | 目标等效带宽 | 说明 |
|---|---|---|
| **快层**（mesh 链路） | **40 GB/s** | 对应论文片内 3.43 TB/s |
| **慢层**（rail 链路） | **5.33 GB/s** | 对应论文 scale-up 460 GB/s，比值 7.5:1 |

### 5.2 为什么要压到远低于硬件能力

A100 NVLink3 单向约 250 GB/s，这里只用 40 GB/s，留 6× 余量。**这是刻意设计**：

- 真实 mesh 的每条链路是**独立专线**，互不争抢；
- 在 NVSwitch 上 emulated link 共享 fabric，**只有当总负载远低于 fabric 能力时，各链路才表现得像独立信道**；
- 因此**压低带宽反而提高保真度**，同时让 7.5:1 这个比值由软件精确控制。

**不要**用 host memory 中转来做慢层——其带宽受 NUMA、pinned 内存、PCIe 竞争影响，不可控。**两层都走 NVLink，纯靠软件限速区分。**

### 5.3 限速实现：分块 + pacing kernel

把一次传输拆成固定 chunk，每个 chunk 之后在**同一个 stream** 上插一个自旋 kernel，补齐到目标耗时。全程在 device 上，无 host 抖动。

**自旋计时必须用 `%globaltimer`，不要用 `clock64()`。**

```cuda
// %globaltimer 直接给纳秒墙上时间，与 SM 频率无关：
//   - 不需要 cudaDeviceProp::clockRate 做周期换算
//   - 不受 DVFS 影响（clock64() 数的是 SM 周期，频率一漂自旋时长就漂）
// 更新粒度几十 ns ~ 1 µs，相对 t_pace ≈ 10^2 µs 可忽略。
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void pace_kernel(unsigned long long ns) {
    unsigned long long t0 = gtimer();
    while (gtimer() - t0 < ns) { }
}
```

> ⚠️ **不要**改回 `clock64()` 版本。`cudaDeviceProp::clockRate` 返回的是**默认 boost 频率**，锁频后不会更新，用它换算 cycles 必错。

**pacing 时长计算**：

```
t_pace_ns(n) = n / B_target  -  n / B_raw  -  O
```

| 符号 | 含义 |
|---|---|
| `n` | **本 chunk 的实际字节数**（不是标称 chunk 大小 `C`） |
| `B_target` | 目标带宽，单位 GB/s（数值上即 字节/ns） |
| `B_raw` | **实测**未限速带宽，由 §7.1 标定模式给出，不能用 spec 值 |
| `O` | **每 chunk 固定开销**（ns），由 §5.4 闭环标定给出 |

**`t_pace` 必须按每个 chunk 的实际 `n` 计算，不能用固定值。** 两个场景会出错：

1. 末尾 chunk 的 `n < C`；
2. **`D = 2 MB` 时 S2/S3 只发 `D/2 = 1 MB`，整个传输就一个 1 MB 的 chunk** —— 若按 4 MB 补自旋，该点等效带宽会低到 1/4。

`O` 是**加性**项而非乘性系数：它来自 `ncclGroupStart/End` 下发（~5–10 µs）、pace kernel launch（~3–5 µs）、stream 上两 kernel 间隙，**与 `n` 无关**。按加性建模，标定出的 `O` 才能跨 chunk 大小复用（含末尾 chunk 与 `D < C` 的情形）。

**chunk 大小**：标称 **4 MB**。4 MB @ 40 GB/s = 105 µs，硬件实际约 16 µs，`O` 约 10 µs，补 ~79 µs。µs 级抖动占比 <1%。

```c
// 返回某个 chunk 该补的自旋纳秒数
static inline unsigned long long pace_ns(size_t n, double B_target,
                                         double B_raw, double O_ns)
{
    double v = (double)n / B_target - (double)n / B_raw - O_ns;
    return (v > 0.0) ? (unsigned long long)v : 0ULL;   // 见下方钳位说明
}

void paced_exchange(void* s, void* r, size_t bytes, int peer,
                    ncclComm_t comm, cudaStream_t st,
                    double B_target, double B_raw, double O_ns)
{
    const size_t C = 4 << 20;
    for (size_t off = 0; off < bytes; off += C) {
        size_t n = (bytes - off < C) ? (bytes - off) : C;
        ncclGroupStart();
        ncclSend((char*)s + off, n, ncclChar, peer, comm, st);
        ncclRecv((char*)r + off, n, ncclChar, peer, comm, st);
        ncclGroupEnd();
        pace_kernel<<<1, 1, 0, st>>>(pace_ns(n, B_target, B_raw, O_ns));
    }
}
```

> **钳位必须告警**：`pace_ns` 算出 ≤0 说明该 chunk 下固定开销已经超过目标耗时，**限速失效，该数据点不可用**。此时必须打印 warning 并在 CSV 里标记该行，不能静默返回 0。



> **AllGather 分块注意**：分块后输出布局变成 chunk-major，不是标准的 `[rank0 | rank1]`。**基准测试只关心耗时，无需还原布局**，用 2C 大小的 scratch 接收即可。

### 5.4 闭环标定（必做）

§5.3 的公式是**开环**的，它只算了传输与自旋，**必然偏慢**（每 chunk 的固定开销只会增加耗时，方向单一）。不做闭环修正的典型误差：

| 项 | 4 MB chunk @ 目标 40 GB/s |
|---|---|
| 目标单 chunk 耗时 | 105 µs |
| 实际传输 | ~16 µs |
| 令 `O = 0` 时的 `t_pace` | 89 µs |
| 未计入的固定开销 | ~10 µs |
| **实际耗时** | **115 µs → B_achieved ≈ 36.5 GB/s（偏低 9%）** |

9% 远超 3% 容差，所以必须迭代标定 `O`：

1. 跑单链路测出 `B_raw`；
2. 令 `O = 0`，按 §5.3 公式跑一次，测实际耗时 `T_measured`；
3. 按下式修正（`N` = chunk 数，`T_target = bytes / B_target`）：
   ```
   O  ←  O + (T_measured - T_target) / N
   ```
   系统接近线性，**一般 1–2 轮进 3%**。实测噪声大时乘 0.7 阻尼防震荡。
4. 迭代到 `|B_achieved − B_target| / B_target ≤ 3%`；
5. **所有报告一律用实测值 `B_achieved`，不用配置值 `B_target`。**

**交叉验证**：标定出的 `O` 在 `D = 2 MB`（单个小 chunk，开销占比最大）与 `D = 256 MB`（64 个满 chunk）两端都必须满足 3% 容差。若两端对不上，说明开销不是纯加性，需要重新建模而不是取折中值。

### 5.5 锁频

#### 为什么必须锁

GPU SM 频率受 DVFS 动态调节（A100 boost 1410 MHz，空闲可掉到 200 MHz 量级）。三个理由，第一个最要紧：

1. **这个 workload 特别容易被降频，而且方向是反的。** `pace_kernel<<<1,1>>>` 只占一个 SM、一个线程，GPU 利用率和功耗接近零，驱动会判定为轻载并**下调频率档**。所以风险不是高负载热节流，而是低负载省电降频。
2. **污染曲线。** 频率漂移带来的是与热状态相关的**系统性偏差**，会在长扫描过程中缓慢漂移，且与 `D` 正相关（大 `D` 跑得久、更热）—— 正好污染最想看的那条曲线。
3. **可复现性。** 要拿 median / p99 与 ASTRA-SIM 比到几个百分点；频率浮动会把方差灌进 p99，使其变成散热指标而非通信指标。

> §5.3 已改用 `%globaltimer`，**自旋时长本身不再受频率影响**，理由 1 的致命性因此解除。锁频在此之后的作用是降低 p99 方差、保证可复现 —— 仍然**必做**，但在无 root 的共享节点上可降级为「记录 + 筛样本」（见下）。

#### 怎么定这个频率

直觉是"选能长期稳住的最高频"，但对这个实验**应该反过来 —— 选最低的可用频率**。

我们本来就把带宽压到硬件能力的 1/6（40 GB/s vs ~250 GB/s），**根本不需要峰值频率**；而低频率意味着永不触功耗墙/温度墙，稳定性最优。唯一约束是降频后 `B_raw` 仍需远高于 `B_target`。

> **判定规则：选满足 `B_raw ≥ 3 × B_target` 的最低支持频率。**

A100 上 **1215 MHz** 是合理起点 —— 此时 NVLink 实测仍在 200 GB/s 以上，对 40 GB/s 有 5× 余量。

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS          # 列出可选值
nvidia-smi -pm 1                           # 持久模式（必须，否则锁定会被重置）
nvidia-smi -lgc 1215,1215                  # 锁 SM 频率
nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active \
           --format=csv -l 1               # 验证确实锁住、无节流
# 跑完恢复
nvidia-smi -rgc
```

#### 自检（不要相信 nvidia-smi 报告锁上了就真锁上了）

用已知 ns 数启动 `pace_kernel`，CUDA event 计时，核对

```
t_measured  ≈  ns_requested        （误差 ≤ 1%）
```

因为已改用 `%globaltimer`，这一项**与锁定频率无关**，不通过说明自旋实现本身有问题。另外独立核对 `clocks.sm` 查询值等于锁定值、`clocks_throttle_reasons.active` 为 `Not Active`。

#### 无 root 时的退路

`-lgc` 需要特权。拿不到时：

- 保持 `%globaltimer` 自旋（已与频率解耦）；
- **每个样本连同 `clocks.sm` 一起记录进 CSV**，事后剔除频率偏离众数的样本；
- 在 FINDINGS.md 中声明未锁频，并报告频率分布。

#### 顺序依赖

**先锁频 → 再标定 → 再测量。** 锁定频率必须写进 `calibration.json`（§7.3 的 `clock_rate_khz` 字段，填**实际锁定值**而非 `clockRate`）。频率一改，之前的标定全部作废。

---

## 6. 参数设置与扫描

### 6.1 数据量 D

```
2 MB, 8 MB, 32 MB, 64 MB, 256 MB
```

分段意义：

| 区间 | 意义 |
|---|---|
| 64M–256M | 与 ASTRA-SIM 已发表验证范围（64 MB–1.5 GB）重叠，用于 sanity check |
| 2M–32M | 对齐论文 Fig.14 扫描范围；**32 MB 是论文 Sec.V-A 声明的工作点** |

显存检查：D=256 MB 时 S1 需 send 256 MB + recv 512 MB，A100 40 GB 充足。

> 全部扫描点均位于**带宽主导区**（测试台拐点约 120 KB）。因此本实验验证的是模型的**带宽项**，延迟项未被检验——见 §10。

### 6.2 带宽比 R：固定为论文值，不扫描

```
R = 7.5   （固定；快层 40 GB/s，慢层 5.33 GB/s）
```

来源：论文片内 3.43 TB/s : scale-up 460 GB/s ≈ 7.46。**只验证论文给出的这一个设计点。**

> ⚠️ **不要自行增加 R 扫描。** R≠7.5 不对应任何真实系统，超出本实验范围。

#### 解析预测

各阶段每向传输量：S1 = D（快层）、S2 = D/2（慢层）、S3 = D/2（快层）。设 `B_f` 为快层带宽、`R = B_f / B_s`：

```
t_S1 = D / B_f
t_S2 = (D/2) / B_s = 0.5R · D / B_f
t_S3 = (D/2) / B_f = 0.5  · D / B_f
```

$$T_{\text{total}} = \frac{D}{B_f}\left(1.5 + 0.5R\right) \;\xrightarrow{\;R=7.5\;}\; 5.25\,\frac{D}{B_f}$$

**各阶段耗时占比**（R=7.5）——这是主要对照量：

| 阶段 | 层 | 相对权重 | 占比 |
|---|---|---|---|
| S1 | 快 | 1.0 | **19%** |
| S2 | 慢 | 3.75 | **71%** |
| S3 | 快 | 0.5 | **10%** |

**代入 B_f = 40 GB/s 的绝对预测**（单对、单阶段 median，单位 µs）：

| D | t_S1 | t_S2 | t_S3 | T_total |
|---:|---:|---:|---:|---:|
| 2 MB | 52 | 197 | 26 | 275 |
| 8 MB | 210 | 786 | 105 | 1101 |
| 32 MB | 839 | 3146 | 419 | 4404 |
| 64 MB | 1678 | 6291 | 839 | 8808 |
| 256 MB | 6711 | 25166 | 3355 | 35232 |

程序应复现这组绝对值与三个阶段占比。**用实测 `B_achieved` 重算预测值后再比对，不要用 40 GB/s 标称值。**

> **验证强度说明**（须在 FINDINGS.md 声明）：标定已从真机取走 `B`、`L_eff` 两个参数，因此剩余自由度有限。真正的对照量是**三个阶段占比**与**随 D 的线性度**，而不是总时间单点对齐 —— 后者在参数已标定的前提下几乎必然成立。
>
> 代价：R=7.5 时 S3 仅占 10%，**S3 建模误差 30% 只体现为总时间 3% 偏差，已接近测量噪声（p50 抖动 2–3%）**。因此 S3 的验证强度本质上很弱 —— 必须报告**分阶段耗时**而非只报总时间，否则 S1/S3 等于未被检验。

### 6.3 并发度扫描（验证链路独立性假设）

对 S1，分别用 **1 对 / 2 对 / 4 对**并发运行，比较每对的耗时：

- 差异 <3% → 链路独立性成立，可作为 emulation 保真度证据
- 差异随并发度上升 → 存在 fabric 竞争，这本身是一个有价值的测量（ASTRA-SIM 也应预测出来）

---

## 7. 程序设计

### 7.1 运行模式

| 模式 | 功能 | GPU 数 |
|---|---|---|
| `calibrate` | 测各对 GPU 的未限速带宽 `B_raw` 与延迟 `L_eff`，输出 ASTRA-SIM 参数 | 2 |
| `stage` | 跑单个阶段（s1 / s2 / s3），单对 | 2 |
| `concurrency` | S1 的 1/2/4 对并发度扫描 | 2 / 4 / 8 |
| `sweep` | 跑完整扫描（D × stage，R 固定 7.5），输出 CSV | 2 |

### 7.2 进程模型

**建议：单进程多 GPU**，使用 `ncclCommInitAll` 在一个进程内初始化多个 device 的 communicator，每个 device 一个 stream，NCCL 调用用 `ncclGroupStart/End` 包裹。

**不要引入 MPI 依赖**——本实验主体只用 2 卡、最多 8 卡且在单节点内，单进程方案更简单、更易调试。

### 7.3 CLI

```
a2a_bench --mode <calibrate|stage|concurrency|sweep>
          --stage <s1|s2|s3>
          --size <bytes>
          --fast-bw <GB/s>      # 默认 40
          --ratio <R>           # 慢层 = fast-bw / R，固定 7.5（论文值）
                                #   保留此 flag 仅供调试；sweep 模式不遍历 R
          --chunk <bytes>       # 默认 4MB
          --iters <N>           # 默认 100
          --warmup <N>          # 默认 20
          --out <file.csv>
          --verify              # 可选，校验数据正确性
```

### 7.4 输出 CSV schema

```csv
mode,stage,pair,data_size_bytes,per_link_bytes,tier,target_bw_gbps,achieved_bw_gbps,ratio_R,chunk_bytes,pace_clamped,sm_clock_mhz,iters,t_mean_us,t_median_us,t_p99_us,t_min_us,t_max_us
```

> `pace_clamped` = 该配置下是否触发了 §5.3 的自旋钳位（`t_pace ≤ 0`）。**为 1 的行限速失效，分析时必须剔除。**

额外输出一份 `calibration.json`，供配置 ASTRA-SIM：

```json
{
  "fast_tier": {
    "B_raw_gbps": 0.0,
    "B_target_gbps": 0.0,
    "B_achieved_gbps": 0.0,
    "latency_us": 0.0,
    "O_ns": 0.0,
    "calib_iters": 0
  },
  "slow_tier": {
    "B_raw_gbps": 0.0,
    "B_target_gbps": 0.0,
    "B_achieved_gbps": 0.0,
    "latency_us": 0.0,
    "O_ns": 0.0,
    "calib_iters": 0
  },
  "chunk_bytes": 4194304,
  "sm_clock_locked_mhz": 0,
  "sm_clock_locked": true,
  "pace_timer": "globaltimer"
}
```

| 字段 | 说明 |
|---|---|
| `O_ns` | §5.4 闭环标定出的每 chunk 固定开销，**必须记录** —— 复跑时直接复用，不重标 |
| `calib_iters` | 标定迭代轮数；>3 轮说明系统非线性，需在 FINDINGS.md 说明 |
| `sm_clock_locked_mhz` | **实际锁定频率**，不是 `cudaDeviceProp::clockRate` |
| `sm_clock_locked` | 是否成功锁频；`false` 时须按 §5.5「无 root 退路」处理 |
| `pace_timer` | 固定为 `globaltimer`；出现 `clock64` 即为实现错误 |

---

## 8. 测量协议

**必须严格遵守，否则数据不可用：**

1. **计时**：CUDA event 夹在 stream 两端。两个 rank **各自计时取 max**。测量开始前插入 barrier（`cudaDeviceSynchronize` 两侧 + 同步点）。
2. **预热**：warmup ≥ **20** 次，不计入统计。
3. **样本数**：正式测量 ≥ **100** 次。
4. **统计量**：报告 **median** 与 **p99**，不要只报 mean。
5. **防 L2 命中**：A100 L2 有 40 MB，较小尺寸（2–8 MB）会被 L2 吃掉导致带宽虚高。
   → **准备一组轮转 buffer，总量 > 80 MB，每次迭代换用不同 buffer。**
6. **锁频**：见 §5.5。按「满足 `B_raw ≥ 3 × B_target` 的最低支持频率」选取（A100 起点 1215 MHz），`nvidia-smi -pm 1 && nvidia-smi -lgc 1215,1215`，并通过 §5.5 自检。**顺序：先锁频 → 再标定 → 再测量。**
7. **绑 NUMA**：`numactl --cpunodebind=<node>`，node 按 `nvidia-smi topo -m` 的亲和性选择。
8. **记录频率**：每个样本连同 `clocks.sm` 一起落盘（CSV 增列 `sm_clock_mhz`），用于事后检测频率漂移并剔除样本。

---

## 9. 自检与验收

程序应内置以下断言/检查：

- [ ] **拓扑合法性**：统计所有 GPU 对之间的通信次数，**非邻居对（0–3, 1–2, 4–7, 5–6）必须为 0**
- [ ] **达成带宽**：各层 `|B_achieved − B_target| / B_target < 3%`，且在 `D = 2 MB` 与 `D = 256 MB` **两端都满足**
- [ ] **自旋实现**：用已知 ns 数单跑 `pace_kernel`，CUDA event 计时误差 ≤ 1%（§5.5 自检）
- [ ] **无钳位**：全部数据点 `pace_clamped == 0`；否则该点限速失效，须剔除并在报告中声明
- [ ] **字节计数**：软件统计的各阶段实际传输字节数，与 §4.4 的解析值逐项一致
- [ ] **拓扑确认**：启动时打印 `nvidia-smi topo -m` 的解析结果，确认 GPU 映射与 §2 一致
- [ ] **解析对照**：三个阶段的耗时占比应接近 §6.2 的 19% / 71% / 10%；各阶段耗时随 D 应为过原点线性
- [ ] **分阶段上报**：CSV 必须含每个阶段的独立计时行。**只报总时间视为未完成** —— 见 §6.2 验证强度说明

### 环境前置检查

程序启动时应检查并报告：

- CUDA / NCCL 版本（NCCL ≥ 2.17）
- GPU 数量 ≥ 2（`concurrency` 模式需 ≥ 8）
- P2P 可达性矩阵
- 持久模式与锁频状态；**实际 `clocks.sm` 是否等于目标锁定值**（不等则拒绝启动 `sweep` 模式，或按 §5.5 退路降级并告警）
- 计算能力 ≥ sm_70（`%globaltimer` 可用）

> **不需要** `cudaDeviceProp::clockRate` —— §5.3 已改为 `%globaltimer` 纳秒计时，不做周期换算。该字段仅可用于信息打印，**不得参与 pacing 计算**。

---

## 10. 已知限制（须在代码注释与报告中声明）

1. **零热点**：2×2 mesh 拓扑上是 4 节点环，边传递，A2A 负载严格均匀。论文论断中"不均衡导致局部热点"这一机制**在本配置下未被检验**。
2. **链路非物理隔离**：emulated link 共享 NVSwitch fabric 与 GPU 注入带宽，靠大余量限速近似独立信道，并非真正的专线。
3. **阶段全退化**：三个阶段均为 2 成员组、单跳，未检验多成员 collective 的调度行为，也不涉及多跳转发。
4. **仅验证带宽项**：扫描范围 2–256 MB 全部位于带宽主导区（测试台拐点约 120 KB），**模型的延迟项 `L_eff` 未被检验**。
5. **单对测量的损失**：拿不到 4 对之间的 straggler / p99，也拿不到跨阶段 overlap。三阶段单对耗时之和是算法总时间的**下界**，不是实测值。
6. **口径一致性**：与 ASTRA-SIM 对照时，两边必须使用相同的测量口径（都用单对下界，或都用全量）。**口径一致比口径"正确"更重要。**

---

## 11. 交付物

1. 可编译运行的基准测试程序（建议 C++/CUDA + NCCL，单文件或小型 CMake 工程）
2. `README.md`：编译方式、依赖版本、运行示例
3. `calibration.json`：链路参数，供配置 ASTRA-SIM
4. `results.csv`：完整扫描结果
5. 一份简短的 `FINDINGS.md`：
   - 各阶段耗时 vs D 曲线（应为过原点线性），与 §6.2 绝对预测表的逐点对照
   - 三个阶段的耗时占比，与 §6.2 的 19% / 71% / 10% 对照
   - §6.2 的验证强度说明（标定已取走 2 个参数；S3 占比仅 10%，其验证强度弱）
   - 若未锁频：频率分布与剔除样本数（§5.5）
