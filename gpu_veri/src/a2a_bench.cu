#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::ostringstream os_;                                                    \
      os_ << "CUDA failure at " << __FILE__ << ':' << __LINE__ << ": "          \
          << cudaGetErrorString(status_);                                        \
      throw std::runtime_error(os_.str());                                       \
    }                                                                            \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status_ = (call);                                               \
    if (status_ != ncclSuccess) {                                                \
      std::ostringstream os_;                                                    \
      os_ << "NCCL failure at " << __FILE__ << ':' << __LINE__ << ": "          \
          << ncclGetErrorString(status_);                                        \
      throw std::runtime_error(os_.str());                                       \
    }                                                                            \
  } while (0)

constexpr size_t kMiB = 1ULL << 20;
constexpr size_t kRotationBytes = 80ULL * kMiB;
constexpr double kPaperRatio = 7.5;
constexpr int kLogicalRanks = 8;

// Scope boundary: this 2x(2x2) emulation validates stage byte counts, stage
// shares, and bandwidth-dominated scaling. It does not reproduce 6x6-mesh
// hotspots, physical link isolation, multi-hop collectives, cross-stage
// overlap, or full-algorithm stragglers. Keep simulator and hardware results
// on the same single-pair, per-stage timing basis.

__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void pace_kernel(unsigned long long ns) {
  const unsigned long long begin = global_timer_ns();
  while (global_timer_ns() - begin < ns) {
  }
}

__global__ void count_mismatches(const unsigned char* data, size_t count,
                                 unsigned char expected,
                                 unsigned long long* errors) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    local += data[i] != expected;
  }
  if (local != 0) atomicAdd(errors, local);
}

enum class Stage { S1, S2, S3 };

std::string stage_name(Stage stage) {
  switch (stage) {
    case Stage::S1: return "s1";
    case Stage::S2: return "s2";
    case Stage::S3: return "s3";
  }
  throw std::logic_error("unreachable stage");
}

Stage parse_stage(const std::string& value) {
  if (value == "s1") return Stage::S1;
  if (value == "s2") return Stage::S2;
  if (value == "s3") return Stage::S3;
  throw std::runtime_error("--stage must be s1, s2, or s3");
}

std::array<int, 2> logical_pair(Stage stage) {
  if (stage == Stage::S1) return {0, 2};
  if (stage == Stage::S2) return {0, 4};
  return {0, 1};
}

std::string pair_name(const std::array<int, 2>& pair) {
  return std::to_string(pair[0]) + "-" + std::to_string(pair[1]);
}

bool legal_pair(Stage stage, int a, int b) {
  if (a > b) std::swap(a, b);
  static const std::array<std::array<int, 2>, 4> y{{{0, 2}, {1, 3}, {4, 6}, {5, 7}}};
  static const std::array<std::array<int, 2>, 4> w{{{0, 4}, {1, 5}, {2, 6}, {3, 7}}};
  static const std::array<std::array<int, 2>, 4> x{{{0, 1}, {2, 3}, {4, 5}, {6, 7}}};
  const auto& pairs = stage == Stage::S1 ? y : (stage == Stage::S2 ? w : x);
  return std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) {
    return p[0] == a && p[1] == b;
  });
}

size_t parse_size(std::string text) {
  if (text.empty()) throw std::runtime_error("empty byte size");
  size_t multiplier = 1;
  const char suffix = static_cast<char>(std::tolower(text.back()));
  if (!std::isdigit(text.back())) {
    text.pop_back();
    if (suffix == 'k') multiplier = 1ULL << 10;
    else if (suffix == 'm') multiplier = 1ULL << 20;
    else if (suffix == 'g') multiplier = 1ULL << 30;
    else throw std::runtime_error("unknown size suffix; use K, M, or G");
  }
  const double value = std::stod(text);
  if (value <= 0 || value * multiplier > std::numeric_limits<size_t>::max())
    throw std::runtime_error("invalid byte size");
  return static_cast<size_t>(value * multiplier);
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) values.push_back(std::stoi(item));
  return values;
}

struct Options {
  std::string mode;
  Stage stage = Stage::S1;
  size_t data_size = 32 * kMiB;
  size_t chunk = 4 * kMiB;
  double fast_bw = 40.0;
  double ratio = kPaperRatio;
  int iters = 100;
  int warmup = 20;
  int latency_iters = 1000;
  std::string out = "results.csv";
  std::string calibration = "calibration.json";
  bool verify = false;
  bool strict_clock = false;
  bool sample_clocks = true;
  bool skip_timer_check = false;
  int expected_clock_mhz = 0;
  std::array<int, kLogicalRanks> device_map{{0, 1, 2, 3, 4, 5, 6, 7}};
};

void usage(std::ostream& out) {
  out << "Mesh-Rail A2A verifier\n\n"
      << "  a2a_bench --mode <calibrate|stage|concurrency|sweep> [options]\n\n"
      << "Core options:\n"
      << "  --stage <s1|s2|s3>       stage mode only\n"
      << "  --size <bytes|2M>         per-rank A2A buffer D (default 32M)\n"
      << "  --fast-bw <GB/s>          target fast-tier bandwidth (default 40)\n"
      << "  --ratio <R>               B_fast/B_slow (default and sweep value 7.5)\n"
      << "  --chunk <bytes|4M>        pacing chunk (default 4M)\n"
      << "  --iters <N>               measured iterations (default 100)\n"
      << "  --warmup <N>              warmup iterations (default 20)\n"
      << "  --out <file.csv>          summary CSV\n"
      << "  --calibration <file.json> generated/loaded calibration\n"
      << "  --verify                   verify received bytes after timing\n\n"
      << "Environment/reproducibility options:\n"
      << "  --device-map <2 or 8 IDs> logical rank mapping; 2 IDs compact non-concurrency modes\n"
      << "  --expected-clock <MHz>    expected locked SM clock\n"
      << "  --strict-clock            fail if sampled clock differs\n"
      << "  --no-sample-clocks        skip per-iteration nvidia-smi sampling\n"
      << "  --skip-timer-check        debugging only; skip %globaltimer self-test\n";
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--mode") opts.mode = value();
    else if (arg == "--stage") opts.stage = parse_stage(value());
    else if (arg == "--size") opts.data_size = parse_size(value());
    else if (arg == "--chunk") opts.chunk = parse_size(value());
    else if (arg == "--fast-bw") opts.fast_bw = std::stod(value());
    else if (arg == "--ratio") opts.ratio = std::stod(value());
    else if (arg == "--iters") opts.iters = std::stoi(value());
    else if (arg == "--warmup") opts.warmup = std::stoi(value());
    else if (arg == "--latency-iters") opts.latency_iters = std::stoi(value());
    else if (arg == "--out") opts.out = value();
    else if (arg == "--calibration") opts.calibration = value();
    else if (arg == "--expected-clock") opts.expected_clock_mhz = std::stoi(value());
    else if (arg == "--device-map") {
      const auto values = parse_int_list(value());
      if (values.size() == 2) {
        opts.device_map.fill(values[1]);
        opts.device_map[0] = values[0];
      } else if (values.size() == kLogicalRanks) {
        std::copy(values.begin(), values.end(), opts.device_map.begin());
      } else {
        throw std::runtime_error("--device-map requires 2 or 8 CUDA device IDs");
      }
    } else if (arg == "--verify") opts.verify = true;
    else if (arg == "--strict-clock") opts.strict_clock = true;
    else if (arg == "--no-sample-clocks") opts.sample_clocks = false;
    else if (arg == "--skip-timer-check") opts.skip_timer_check = true;
    else if (arg == "--help" || arg == "-h") {
      usage(std::cout);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  if (opts.mode != "calibrate" && opts.mode != "stage" &&
      opts.mode != "concurrency" && opts.mode != "sweep")
    throw std::runtime_error("--mode must be calibrate, stage, concurrency, or sweep");
  if (opts.data_size == 0 || opts.chunk == 0 || opts.fast_bw <= 0 ||
      opts.ratio <= 0 || opts.iters < 1 || opts.warmup < 0 ||
      opts.latency_iters < 1)
    throw std::runtime_error("sizes, bandwidth, ratio, and iteration counts must be positive");
  if (opts.mode == "sweep" && std::abs(opts.ratio - kPaperRatio) > 1e-12)
    throw std::runtime_error("sweep is fixed to the paper ratio R=7.5; use stage mode for debugging another ratio");
  if (opts.mode == "concurrency") {
    std::vector<int> unique(opts.device_map.begin(), opts.device_map.end());
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.size() != kLogicalRanks)
      throw std::runtime_error("concurrency mode requires 8 distinct entries in --device-map");
  }
  return opts;
}

std::string shell_output(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string result;
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return result;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) result += buffer.data();
  const int status = pclose(pipe);
  if (status != 0) return {};
  return result;
}

std::vector<int> query_sm_clocks(const std::vector<int>& devices) {
  std::ostringstream ids;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (i) ids << ',';
    ids << devices[i];
  }
  const std::string output = shell_output(
      "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i " + ids.str() + " 2>/dev/null");
  std::vector<int> clocks;
  std::stringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    try { clocks.push_back(std::stoi(line)); } catch (...) { clocks.push_back(0); }
  }
  if (clocks.size() != devices.size()) clocks.assign(devices.size(), 0);
  return clocks;
}

std::vector<int> unique_devices(const std::vector<std::array<int, 2>>& pairs) {
  std::vector<int> devices;
  for (const auto& pair : pairs) {
    devices.push_back(pair[0]);
    devices.push_back(pair[1]);
  }
  std::sort(devices.begin(), devices.end());
  devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
  return devices;
}

void print_environment(const Options& opts) {
  int runtime = 0, driver = 0, nccl = 0, count = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  NCCL_CHECK(ncclGetVersion(&nccl));
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::cout << "CUDA runtime=" << runtime << " driver=" << driver
            << " NCCL=" << nccl << " visible_gpus=" << count << '\n';
  if (nccl < 21700) throw std::runtime_error("NCCL >= 2.17 is required");

  std::vector<int> mapped(opts.device_map.begin(), opts.device_map.end());
  std::cout << "Logical-to-physical device map:";
  for (int logical = 0; logical < kLogicalRanks; ++logical)
    std::cout << ' ' << logical << "->" << opts.device_map[logical];
  std::cout << '\n';
  for (int device : mapped) {
    if (device < 0 || device >= count)
      throw std::runtime_error("--device-map references a CUDA device that is not visible");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7)
      throw std::runtime_error("logical mapping includes a GPU older than sm_70");
  }

  std::cout << "P2P reachability for visible CUDA devices:\n    ";
  for (int j = 0; j < count; ++j) std::cout << std::setw(2) << j << ' ';
  std::cout << '\n';
  for (int i = 0; i < count; ++i) {
    std::cout << std::setw(2) << i << ": ";
    for (int j = 0; j < count; ++j) {
      int access = i == j;
      if (i != j) CUDA_CHECK(cudaDeviceCanAccessPeer(&access, i, j));
      std::cout << std::setw(2) << access << ' ';
    }
    std::cout << '\n';
  }

  std::vector<std::array<int, 2>> required;
  if (opts.mode == "concurrency") {
    required = {{{0, 2}, {1, 3}, {4, 6}, {5, 7}}};
  } else if (opts.mode == "stage") {
    required = {logical_pair(opts.stage)};
  } else {
    required = {logical_pair(Stage::S1), logical_pair(Stage::S2),
                logical_pair(Stage::S3)};
  }
  for (const auto& logical : required) {
    const int a = opts.device_map[logical[0]], b = opts.device_map[logical[1]];
    int ab = 0, ba = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&ab, a, b));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&ba, b, a));
    if (!ab || !ba)
      throw std::runtime_error("required logical pair " + pair_name(logical) +
                               " is not bidirectionally P2P reachable");
  }

  std::string state = shell_output(
      "nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,"
      "clocks_event_reasons.active --format=csv 2>/dev/null");
  if (state.empty()) {
    state = shell_output(
        "nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,"
        "clocks_throttle_reasons.active --format=csv 2>/dev/null");
  }
  if (!state.empty()) std::cout << "GPU clock/persistence state:\n" << state;
  const std::string topo = shell_output("nvidia-smi topo -m 2>/dev/null");
  if (topo.empty()) {
    std::cerr << "WARNING: could not obtain `nvidia-smi topo -m`; preserve it separately.\n";
  } else {
    std::cout << "nvidia-smi topology:\n" << topo;
  }
}

struct DeviceBuffer {
  unsigned char* send = nullptr;
  unsigned char* recv = nullptr;
  unsigned long long* errors = nullptr;
};

struct PairContext {
  std::array<int, 2> logical{};
  std::array<int, 2> physical{};
  std::array<ncclComm_t, 2> comm{{nullptr, nullptr}};
  std::array<cudaStream_t, 2> stream{{nullptr, nullptr}};
  std::array<cudaEvent_t, 2> begin{{nullptr, nullptr}};
  std::array<cudaEvent_t, 2> end{{nullptr, nullptr}};
  std::array<DeviceBuffer, 2> buffers{};
  size_t send_stride = 0;
  size_t recv_stride = 0;
  size_t slots = 0;

  PairContext(std::array<int, 2> logical_pair,
              const std::array<int, kLogicalRanks>& mapping)
      : logical(logical_pair), physical{{mapping[logical[0]], mapping[logical[1]]}} {
    if (physical[0] == physical[1]) throw std::runtime_error("pair maps both ranks to one GPU");
    NCCL_CHECK(ncclCommInitAll(comm.data(), 2, physical.data()));
    for (int rank = 0; rank < 2; ++rank) {
      CUDA_CHECK(cudaSetDevice(physical[rank]));
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream[rank], cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreate(&begin[rank]));
      CUDA_CHECK(cudaEventCreate(&end[rank]));
    }
  }

  PairContext(const PairContext&) = delete;
  PairContext& operator=(const PairContext&) = delete;

  void allocate(Stage stage, size_t transfer_bytes, bool rotate = true) {
    release_buffers();
    send_stride = transfer_bytes;
    recv_stride = stage == Stage::S1 ? 2 * transfer_bytes : transfer_bytes;
    slots = rotate ? std::max<size_t>(2, (kRotationBytes + transfer_bytes) / transfer_bytes) : 1;
    for (int rank = 0; rank < 2; ++rank) {
      CUDA_CHECK(cudaSetDevice(physical[rank]));
      CUDA_CHECK(cudaMalloc(&buffers[rank].send, send_stride * slots));
      CUDA_CHECK(cudaMalloc(&buffers[rank].recv, recv_stride * slots));
      CUDA_CHECK(cudaMalloc(&buffers[rank].errors, sizeof(unsigned long long)));
      const unsigned char pattern = rank == 0 ? 0x35 : 0xa7;
      CUDA_CHECK(cudaMemsetAsync(buffers[rank].send, pattern, send_stride * slots, stream[rank]));
      CUDA_CHECK(cudaMemsetAsync(buffers[rank].recv, 0, recv_stride * slots, stream[rank]));
      CUDA_CHECK(cudaMemsetAsync(buffers[rank].errors, 0, sizeof(unsigned long long), stream[rank]));
    }
    synchronize();
  }

  void synchronize() const {
    for (int rank = 0; rank < 2; ++rank) {
      CUDA_CHECK(cudaSetDevice(physical[rank]));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void release_buffers() noexcept {
    for (int rank = 0; rank < 2; ++rank) {
      if (physical[rank] >= 0) cudaSetDevice(physical[rank]);
      if (buffers[rank].send) cudaFree(buffers[rank].send);
      if (buffers[rank].recv) cudaFree(buffers[rank].recv);
      if (buffers[rank].errors) cudaFree(buffers[rank].errors);
      buffers[rank] = {};
    }
  }

  ~PairContext() {
    release_buffers();
    for (int rank = 0; rank < 2; ++rank) {
      cudaSetDevice(physical[rank]);
      if (begin[rank]) cudaEventDestroy(begin[rank]);
      if (end[rank]) cudaEventDestroy(end[rank]);
      if (stream[rank]) cudaStreamDestroy(stream[rank]);
      if (comm[rank]) ncclCommDestroy(comm[rank]);
    }
  }
};

struct PaceSpec {
  bool enabled = false;
  double target_gbps = 0;
  double raw_gbps = 0;
  double overhead_ns = 0;
};

struct PaceValue {
  unsigned long long ns = 0;
  bool clamped = false;
};

PaceValue pace_value(size_t bytes, const PaceSpec& pace) {
  if (!pace.enabled) return {};
  if (pace.target_gbps <= 0 || pace.raw_gbps <= pace.target_gbps)
    throw std::runtime_error("pacing requires B_raw > B_target > 0");
  const double value = static_cast<double>(bytes) / pace.target_gbps -
                       static_cast<double>(bytes) / pace.raw_gbps -
                       pace.overhead_ns;
  if (value <= 0) return {0, true};
  return {static_cast<unsigned long long>(std::llround(value)), false};
}

bool enqueue_chunk(const std::vector<PairContext*>& contexts, Stage stage,
                   size_t slot, size_t offset, size_t bytes,
                   const PaceSpec& pace) {
  bool clamped = false;
  NCCL_CHECK(ncclGroupStart());
  for (PairContext* ctx : contexts) {
    for (int rank = 0; rank < 2; ++rank) {
      CUDA_CHECK(cudaSetDevice(ctx->physical[rank]));
      unsigned char* send = ctx->buffers[rank].send + slot * ctx->send_stride + offset;
      if (stage == Stage::S1) {
        unsigned char* recv = ctx->buffers[rank].recv + slot * ctx->recv_stride + 2 * offset;
        NCCL_CHECK(ncclAllGather(send, recv, bytes, ncclChar,
                                 ctx->comm[rank], ctx->stream[rank]));
      } else {
        unsigned char* recv = ctx->buffers[rank].recv + slot * ctx->recv_stride + offset;
        NCCL_CHECK(ncclSend(send, bytes, ncclChar, 1 - rank,
                            ctx->comm[rank], ctx->stream[rank]));
        NCCL_CHECK(ncclRecv(recv, bytes, ncclChar, 1 - rank,
                            ctx->comm[rank], ctx->stream[rank]));
      }
    }
  }
  NCCL_CHECK(ncclGroupEnd());

  const PaceValue delay = pace_value(bytes, pace);
  clamped |= delay.clamped;
  if (pace.enabled) {
    for (PairContext* ctx : contexts) {
      for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(ctx->physical[rank]));
        pace_kernel<<<1, 1, 0, ctx->stream[rank]>>>(delay.ns);
        CUDA_CHECK(cudaGetLastError());
      }
    }
  }
  return clamped;
}

bool enqueue_operation(const std::vector<PairContext*>& contexts, Stage stage,
                       size_t slot, size_t transfer_bytes, size_t chunk,
                       const PaceSpec& pace) {
  bool clamped = false;
  for (size_t offset = 0; offset < transfer_bytes; offset += chunk) {
    const size_t actual = std::min(chunk, transfer_bytes - offset);
    clamped |= enqueue_chunk(contexts, stage, slot, offset, actual, pace);
  }
  return clamped;
}

struct IterationSample {
  std::vector<std::array<double, 2>> endpoint_us;
  std::vector<int> clocks_mhz;
};

struct RunResult {
  std::vector<std::vector<double>> pair_times_us;
  std::vector<IterationSample> samples;
  std::vector<int> clock_devices;
  bool pace_clamped = false;
};

RunResult run_contexts(const std::vector<PairContext*>& contexts, Stage stage,
                       size_t transfer_bytes, size_t chunk, const PaceSpec& pace,
                       int warmup, int iters, bool sample_clocks) {
  if (contexts.empty()) throw std::runtime_error("no communication pairs");
  for (PairContext* ctx : contexts) {
    if (ctx->send_stride != transfer_bytes)
      throw std::logic_error("buffer stride does not match transfer size");
  }
  const size_t slots = contexts.front()->slots;
  for (int iter = 0; iter < warmup; ++iter) {
    enqueue_operation(contexts, stage, iter % slots, transfer_bytes, chunk, pace);
  }
  for (PairContext* ctx : contexts) ctx->synchronize();

  RunResult result;
  result.pair_times_us.resize(contexts.size());
  const auto physical = unique_devices([&] {
    std::vector<std::array<int, 2>> p;
    for (PairContext* ctx : contexts) p.push_back(ctx->physical);
    return p;
  }());
  result.clock_devices = physical;

  for (int iter = 0; iter < iters; ++iter) {
    for (PairContext* ctx : contexts) ctx->synchronize();  // iteration barrier
    for (PairContext* ctx : contexts) {
      for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(ctx->physical[rank]));
        CUDA_CHECK(cudaEventRecord(ctx->begin[rank], ctx->stream[rank]));
      }
    }
    result.pace_clamped |= enqueue_operation(
        contexts, stage, (warmup + iter) % slots, transfer_bytes, chunk, pace);
    for (PairContext* ctx : contexts) {
      for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(ctx->physical[rank]));
        CUDA_CHECK(cudaEventRecord(ctx->end[rank], ctx->stream[rank]));
      }
    }

    IterationSample sample;
    sample.endpoint_us.resize(contexts.size());
    for (size_t p = 0; p < contexts.size(); ++p) {
      PairContext* ctx = contexts[p];
      std::array<double, 2> endpoint{};
      for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(ctx->physical[rank]));
        CUDA_CHECK(cudaEventSynchronize(ctx->end[rank]));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->begin[rank], ctx->end[rank]));
        endpoint[rank] = static_cast<double>(ms) * 1000.0;
      }
      sample.endpoint_us[p] = endpoint;
      result.pair_times_us[p].push_back(std::max(endpoint[0], endpoint[1]));
    }
    if (sample_clocks) sample.clocks_mhz = query_sm_clocks(physical);
    result.samples.push_back(std::move(sample));
  }
  if (result.pace_clamped)
    std::cerr << "WARNING: t_pace <= 0 for at least one chunk; result is invalid.\n";

  // Runtime traffic ledger. Only the listed stage-neighbor pairs receive
  // nonzero counts, so forbidden diagonal traffic is asserted to stay zero.
  std::array<std::array<unsigned long long, kLogicalRanks>, kLogicalRanks> ledger{};
  const unsigned long long invocations = static_cast<unsigned long long>(warmup + iters);
  for (PairContext* ctx : contexts) {
    const int a = ctx->logical[0], b = ctx->logical[1];
    ledger[a][b] += invocations;
    ledger[b][a] += invocations;
    std::cout << "Traffic audit: " << stage_name(stage) << " pair "
              << pair_name(ctx->logical) << ", calls/direction=" << invocations
              << ", bytes/direction=" << invocations * transfer_bytes << '\n';
  }
  for (int a = 0; a < kLogicalRanks; ++a) {
    for (int b = a + 1; b < kLogicalRanks; ++b) {
      if (!legal_pair(stage, a, b) && (ledger[a][b] != 0 || ledger[b][a] != 0))
        throw std::logic_error("runtime traffic ledger contains a non-neighbor communication");
    }
  }
  return result;
}

void verify_result(PairContext& ctx, Stage stage, size_t transfer_bytes,
                   size_t chunk, const PaceSpec& pace) {
  enqueue_operation({&ctx}, stage, 0, transfer_bytes, chunk, pace);
  ctx.synchronize();
  constexpr int threads = 256;
  constexpr int blocks = 256;
  for (int receiver = 0; receiver < 2; ++receiver) {
    CUDA_CHECK(cudaSetDevice(ctx.physical[receiver]));
    CUDA_CHECK(cudaMemset(ctx.buffers[receiver].errors, 0, sizeof(unsigned long long)));
    if (stage == Stage::S1) {
      for (size_t offset = 0; offset < transfer_bytes; offset += chunk) {
        const size_t actual = std::min(chunk, transfer_bytes - offset);
        for (int source = 0; source < 2; ++source) {
          const unsigned char expected = source == 0 ? 0x35 : 0xa7;
          const unsigned char* ptr = ctx.buffers[receiver].recv + 2 * offset + source * actual;
          count_mismatches<<<blocks, threads>>>(ptr, actual, expected,
                                                ctx.buffers[receiver].errors);
        }
      }
    } else {
      const unsigned char expected = receiver == 0 ? 0xa7 : 0x35;
      count_mismatches<<<blocks, threads>>>(ctx.buffers[receiver].recv,
                                            transfer_bytes, expected,
                                            ctx.buffers[receiver].errors);
    }
    CUDA_CHECK(cudaGetLastError());
    unsigned long long errors = 0;
    CUDA_CHECK(cudaMemcpy(&errors, ctx.buffers[receiver].errors,
                          sizeof(errors), cudaMemcpyDeviceToHost));
    if (errors != 0)
      throw std::runtime_error("data verification failed with " +
                               std::to_string(errors) + " mismatched bytes");
  }
  std::cout << "Data verification passed for " << stage_name(stage) << ' '
            << pair_name(ctx.logical) << ".\n";
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  if (q == 0.5 && values.size() % 2 == 0) {
    const size_t upper = values.size() / 2;
    return (values[upper - 1] + values[upper]) / 2.0;
  }
  const size_t index = static_cast<size_t>(std::ceil(q * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

struct Stats {
  double mean = 0, median = 0, p99 = 0, min = 0, max = 0;
};

Stats summarize(const std::vector<double>& values) {
  if (values.empty()) return {};
  Stats stats;
  stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  stats.median = percentile(values, 0.5);
  stats.p99 = percentile(values, 0.99);
  stats.min = *std::min_element(values.begin(), values.end());
  stats.max = *std::max_element(values.begin(), values.end());
  return stats;
}

struct CsvRow {
  std::string mode, stage, pair, tier;
  size_t data_size = 0, per_link_bytes = 0, chunk = 0;
  double target_bw = 0, achieved_bw = 0, ratio = 0;
  bool clamped = false;
  int sm_clock = 0, iters = 0;
  Stats time;
};

void ensure_parent(const std::string& path) {
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) fs::create_directories(parent);
}

void write_summary(const std::string& path, const std::vector<CsvRow>& rows) {
  ensure_parent(path);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "mode,stage,pair,data_size_bytes,per_link_bytes,tier,target_bw_gbps,"
         "achieved_bw_gbps,ratio_R,chunk_bytes,pace_clamped,sm_clock_mhz,iters,"
         "t_mean_us,t_median_us,t_p99_us,t_min_us,t_max_us\n";
  out << std::fixed << std::setprecision(6);
  for (const auto& row : rows) {
    out << row.mode << ',' << row.stage << ',' << row.pair << ','
        << row.data_size << ',' << row.per_link_bytes << ',' << row.tier << ','
        << row.target_bw << ',' << row.achieved_bw << ',' << row.ratio << ','
        << row.chunk << ',' << (row.clamped ? 1 : 0) << ',' << row.sm_clock << ','
        << row.iters << ',' << row.time.mean << ',' << row.time.median << ','
        << row.time.p99 << ',' << row.time.min << ',' << row.time.max << '\n';
  }
}

std::string samples_path(const std::string& summary_path) {
  fs::path path(summary_path);
  if (path.extension() == ".csv") path.replace_extension(".samples.csv");
  else path += ".samples.csv";
  return path.string();
}

class SampleWriter {
 public:
  explicit SampleWriter(const std::string& summary_path) : path_(samples_path(summary_path)) {
    ensure_parent(path_);
    out_.open(path_);
    if (!out_) throw std::runtime_error("cannot write " + path_);
    out_ << "mode,stage,pair,data_size_bytes,iteration,endpoint0_us,endpoint1_us,"
            "pair_max_us,physical_gpu0,physical_gpu1,sm_clock_gpu0_mhz,"
            "sm_clock_gpu1_mhz,pace_clamped\n";
    out_ << std::fixed << std::setprecision(6);
  }

  void add(const std::string& mode, Stage stage, size_t data_size,
           const std::vector<PairContext*>& contexts, const RunResult& result) {
    const auto& devices = result.clock_devices;
    for (size_t iter = 0; iter < result.samples.size(); ++iter) {
      for (size_t p = 0; p < contexts.size(); ++p) {
        const auto& ctx = *contexts[p];
        auto clock_for = [&](int device) {
          const auto it = std::find(devices.begin(), devices.end(), device);
          if (it == devices.end() || result.samples[iter].clocks_mhz.empty()) return 0;
          return result.samples[iter].clocks_mhz[std::distance(devices.begin(), it)];
        };
        const auto endpoint = result.samples[iter].endpoint_us[p];
        out_ << mode << ',' << stage_name(stage) << ',' << pair_name(ctx.logical)
             << ',' << data_size << ',' << iter << ',' << endpoint[0] << ','
             << endpoint[1] << ',' << std::max(endpoint[0], endpoint[1]) << ','
             << ctx.physical[0] << ',' << ctx.physical[1] << ','
             << clock_for(ctx.physical[0]) << ',' << clock_for(ctx.physical[1])
             << ',' << (result.pace_clamped ? 1 : 0) << '\n';
      }
    }
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::ofstream out_;
};

std::array<int, 2> modal_clocks_for_pair(const PairContext& ctx,
                                         const RunResult& result) {
  std::array<int, 2> modes{{0, 0}};
  for (int rank = 0; rank < 2; ++rank) {
    const auto device_it = std::find(result.clock_devices.begin(),
                                     result.clock_devices.end(),
                                     ctx.physical[rank]);
    if (device_it == result.clock_devices.end()) continue;
    const size_t clock_index = std::distance(result.clock_devices.begin(), device_it);
    std::map<int, size_t> counts;
    for (const auto& sample : result.samples) {
      if (clock_index < sample.clocks_mhz.size() && sample.clocks_mhz[clock_index] > 0)
        ++counts[sample.clocks_mhz[clock_index]];
    }
    size_t best_count = 0;
    for (const auto& [clock, count] : counts) {
      if (count > best_count) {
        modes[rank] = clock;
        best_count = count;
      }
    }
  }
  return modes;
}

std::vector<double> mode_filtered_times(const PairContext& ctx,
                                        const RunResult& result,
                                        size_t pair_index) {
  const auto modes = modal_clocks_for_pair(ctx, result);
  if (modes[0] == 0 && modes[1] == 0)
    return result.pair_times_us.at(pair_index);

  std::array<size_t, 2> clock_index{};
  for (int rank = 0; rank < 2; ++rank) {
    const auto it = std::find(result.clock_devices.begin(), result.clock_devices.end(),
                              ctx.physical[rank]);
    if (it == result.clock_devices.end())
      throw std::logic_error("sampled clock device mapping is incomplete");
    clock_index[rank] = std::distance(result.clock_devices.begin(), it);
  }

  std::vector<double> kept;
  const auto& times = result.pair_times_us.at(pair_index);
  for (size_t i = 0; i < times.size(); ++i) {
    bool matches = true;
    for (int rank = 0; rank < 2; ++rank) {
      if (modes[rank] == 0) continue;
      if (clock_index[rank] >= result.samples[i].clocks_mhz.size() ||
          result.samples[i].clocks_mhz[clock_index[rank]] != modes[rank])
        matches = false;
    }
    if (matches) kept.push_back(times[i]);
  }
  if (kept.empty())
    throw std::runtime_error("frequency-mode filtering rejected every timing sample");
  return kept;
}

CsvRow make_row(const std::string& mode, Stage stage, size_t data_size,
                size_t transfer_bytes, size_t chunk, double target_bw,
                double ratio, const PairContext& ctx, const RunResult& result,
                size_t pair_index) {
  CsvRow row;
  row.mode = mode;
  row.stage = stage_name(stage);
  row.pair = pair_name(ctx.logical);
  row.data_size = data_size;
  row.per_link_bytes = transfer_bytes;
  row.tier = stage == Stage::S2 ? "slow" : "fast";
  row.target_bw = target_bw;
  row.ratio = ratio;
  row.chunk = chunk;
  row.clamped = result.pace_clamped;
  const auto modes = modal_clocks_for_pair(ctx, result);
  row.sm_clock = modes[0] == 0 ? modes[1]
                 : modes[1] == 0 ? modes[0]
                                 : std::min(modes[0], modes[1]);
  const auto filtered = mode_filtered_times(ctx, result, pair_index);
  row.iters = static_cast<int>(filtered.size());
  row.time = summarize(filtered);
  row.achieved_bw = transfer_bytes / (row.time.median * 1000.0);
  const size_t rejected = result.pair_times_us[pair_index].size() - filtered.size();
  if (rejected != 0)
    std::cerr << "WARNING: frequency-mode filter rejected " << rejected << '/'
              << result.pair_times_us[pair_index].size() << " samples for "
              << stage_name(stage) << ' ' << pair_name(ctx.logical) << ".\n";
  return row;
}

struct TierCalibration {
  double raw = 0, target = 0, achieved = 0, latency_us = 0, overhead_ns = 0;
  int rounds = 0;
  std::array<double, 2> endpoint_achieved{{0, 0}};
};

struct Calibration {
  TierCalibration fast, slow;
  size_t chunk = 0;
  int sm_clock_mhz = 0;
  bool sm_clock_locked = false;
};

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read calibration file " + path);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string object_body(const std::string& json, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  const size_t key_pos = json.find(marker);
  if (key_pos == std::string::npos) throw std::runtime_error("missing JSON object " + key);
  const size_t begin = json.find('{', key_pos + marker.size());
  if (begin == std::string::npos) throw std::runtime_error("malformed JSON object " + key);
  int depth = 0;
  for (size_t i = begin; i < json.size(); ++i) {
    if (json[i] == '{') ++depth;
    else if (json[i] == '}' && --depth == 0) return json.substr(begin, i - begin + 1);
  }
  throw std::runtime_error("unterminated JSON object " + key);
}

double json_number(const std::string& body, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(body, match, pattern)) throw std::runtime_error("missing JSON number " + key);
  return std::stod(match[1]);
}

bool json_bool(const std::string& body, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(body, match, pattern)) throw std::runtime_error("missing JSON bool " + key);
  return match[1] == "true";
}

Calibration load_calibration(const std::string& path) {
  const std::string json = read_file(path);
  Calibration cal;
  auto load_tier = [&](const std::string& name, TierCalibration& tier) {
    const std::string body = object_body(json, name);
    tier.raw = json_number(body, "B_raw_gbps");
    tier.target = json_number(body, "B_target_gbps");
    tier.achieved = json_number(body, "B_achieved_gbps");
    tier.latency_us = json_number(body, "latency_us");
    tier.overhead_ns = json_number(body, "O_ns");
    tier.rounds = static_cast<int>(json_number(body, "calib_iters"));
    const std::string cross = object_body(body, "cross_validation");
    tier.endpoint_achieved[0] = json_number(cross, "D_2MiB_achieved_gbps");
    tier.endpoint_achieved[1] = json_number(cross, "D_256MiB_achieved_gbps");
  };
  load_tier("fast_tier", cal.fast);
  load_tier("slow_tier", cal.slow);
  cal.chunk = static_cast<size_t>(json_number(json, "chunk_bytes"));
  cal.sm_clock_mhz = static_cast<int>(json_number(json, "sm_clock_locked_mhz"));
  cal.sm_clock_locked = json_bool(json, "sm_clock_locked");
  if (json.find("\"pace_timer\": \"globaltimer\"") == std::string::npos)
    throw std::runtime_error("calibration pace_timer is not globaltimer");
  return cal;
}

void write_calibration(const std::string& path, const Calibration& cal) {
  ensure_parent(path);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  auto tier = [&](const char* name, const TierCalibration& value, bool comma) {
    out << "  \"" << name << "\": {\n"
        << "    \"B_raw_gbps\": " << value.raw << ",\n"
        << "    \"B_target_gbps\": " << value.target << ",\n"
        << "    \"B_achieved_gbps\": " << value.achieved << ",\n"
        << "    \"latency_us\": " << value.latency_us << ",\n"
        << "    \"O_ns\": " << value.overhead_ns << ",\n"
        << "    \"calib_iters\": " << value.rounds << ",\n"
        << "    \"cross_validation\": {\n"
        << "      \"D_2MiB_achieved_gbps\": " << value.endpoint_achieved[0] << ",\n"
        << "      \"D_256MiB_achieved_gbps\": " << value.endpoint_achieved[1] << "\n"
        << "    }\n  }" << (comma ? "," : "") << '\n';
  };
  out << std::fixed << std::setprecision(6) << "{\n";
  tier("fast_tier", cal.fast, true);
  tier("slow_tier", cal.slow, true);
  out << "  \"chunk_bytes\": " << cal.chunk << ",\n"
      << "  \"sm_clock_locked_mhz\": " << cal.sm_clock_mhz << ",\n"
      << "  \"sm_clock_locked\": " << (cal.sm_clock_locked ? "true" : "false") << ",\n"
      << "  \"pace_timer\": \"globaltimer\"\n}\n";
}

double timer_self_test(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  cudaStream_t stream{};
  cudaEvent_t begin{}, end{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  constexpr unsigned long long requested_ns = 10'000'000;
  std::vector<double> errors;
  for (int i = 0; i < 7; ++i) {
    CUDA_CHECK(cudaEventRecord(begin, stream));
    pace_kernel<<<1, 1, 0, stream>>>(requested_ns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    errors.push_back(std::abs(ms * 1e6 - requested_ns) / requested_ns);
  }
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return percentile(errors, 0.5);
}

size_t transfer_size(Stage stage, size_t data_size) {
  if ((stage == Stage::S2 || stage == Stage::S3) && data_size % 2 != 0)
    throw std::runtime_error("D must be even for s2/s3");
  return stage == Stage::S1 ? data_size : data_size / 2;
}

TierCalibration calibrate_tier(const Options& opts, Stage stage, double target,
                               SampleWriter& samples, std::vector<CsvRow>& rows) {
  const auto logical = logical_pair(stage);
  if (!legal_pair(stage, logical[0], logical[1])) throw std::logic_error("illegal calibration pair");
  PairContext ctx(logical, opts.device_map);
  const size_t transfer = transfer_size(stage, opts.data_size);
  ctx.allocate(stage, transfer);

  const RunResult raw_result = run_contexts({&ctx}, stage, transfer, opts.chunk,
                                             {}, opts.warmup, opts.iters,
                                             opts.sample_clocks);
  const Stats raw_stats = summarize(mode_filtered_times(ctx, raw_result, 0));
  TierCalibration cal;
  cal.raw = transfer / (raw_stats.median * 1000.0);
  cal.target = target;
  if (cal.raw < 3.0 * target) {
    std::ostringstream message;
    message << "B_raw=" << cal.raw << " GB/s is below 3*B_target=" << 3 * target
            << " GB/s for " << stage_name(stage);
    throw std::runtime_error(message.str());
  }

  // Effective paired-operation latency. It is recorded for ASTRA-SIM but the
  // 2--256 MiB sweep intentionally validates only the bandwidth-dominated term.
  ctx.allocate(stage, 1, false);
  const RunResult latency = run_contexts({&ctx}, stage, 1, 1, {},
                                          std::min(opts.warmup, 20),
                                          opts.latency_iters, false);
  cal.latency_us = summarize(latency.pair_times_us[0]).median;

  ctx.allocate(stage, transfer);
  double overhead = 0;
  RunResult tuned;
  constexpr int max_rounds = 10;
  for (int round = 1; round <= max_rounds; ++round) {
    PaceSpec pace{true, target, cal.raw, overhead};
    tuned = run_contexts({&ctx}, stage, transfer, opts.chunk, pace,
                         opts.warmup, opts.iters, opts.sample_clocks);
    const double measured_us = summarize(mode_filtered_times(ctx, tuned, 0)).median;
    cal.achieved = transfer / (measured_us * 1000.0);
    cal.rounds = round;
    const double relative = std::abs(cal.achieved - target) / target;
    std::cout << stage_name(stage) << " calibration round " << round
              << ": O=" << overhead << " ns, achieved=" << cal.achieved
              << " GB/s, error=" << relative * 100 << "%\n";
    if (relative <= 0.03 && !tuned.pace_clamped) break;
    const double chunks = static_cast<double>((transfer + opts.chunk - 1) / opts.chunk);
    const double target_us = transfer / (target * 1000.0);
    const double correction_ns = (measured_us - target_us) * 1000.0 / chunks;
    overhead = std::max(0.0, overhead + 0.7 * correction_ns);
    if (round == max_rounds)
      throw std::runtime_error("closed-loop calibration did not converge within 10 rounds");
  }
  cal.overhead_ns = overhead;
  samples.add("calibrate", stage, opts.data_size, {&ctx}, tuned);
  rows.push_back(make_row("calibrate", stage, opts.data_size, transfer, opts.chunk,
                          target, opts.ratio, ctx, tuned, 0));

  // Required endpoint cross-validation. O is never retuned here.
  const std::array<size_t, 2> endpoint_d{{2 * kMiB, 256 * kMiB}};
  for (size_t i = 0; i < endpoint_d.size(); ++i) {
    const size_t endpoint_transfer = transfer_size(stage, endpoint_d[i]);
    ctx.allocate(stage, endpoint_transfer);
    const PaceSpec pace{true, target, cal.raw, cal.overhead_ns};
    const RunResult result = run_contexts({&ctx}, stage, endpoint_transfer,
                                           opts.chunk, pace, opts.warmup,
                                           opts.iters, opts.sample_clocks);
    const Stats stats = summarize(mode_filtered_times(ctx, result, 0));
    cal.endpoint_achieved[i] = endpoint_transfer / (stats.median * 1000.0);
    samples.add("calibrate-xval", stage, endpoint_d[i], {&ctx}, result);
    rows.push_back(make_row("calibrate-xval", stage, endpoint_d[i], endpoint_transfer,
                            opts.chunk, target, opts.ratio, ctx, result, 0));
    const double error = std::abs(cal.endpoint_achieved[i] - target) / target;
    if (result.pace_clamped || error > 0.03) {
      std::cerr << "WARNING: " << stage_name(stage) << " endpoint D="
                << endpoint_d[i] << " failed 3% cross-validation (achieved "
                << cal.endpoint_achieved[i] << " GB/s). The calibration will be "
                   "written for diagnosis but must not be used.\n";
    }
  }
  if (opts.verify) {
    ctx.allocate(stage, transfer);
    verify_result(ctx, stage, transfer, opts.chunk,
                  PaceSpec{true, target, cal.raw, cal.overhead_ns});
  }
  return cal;
}

void enforce_clock(const Options& opts, const Calibration* cal = nullptr) {
  std::vector<int> requested;
  for (int logical = 0; logical < kLogicalRanks; ++logical)
    requested.push_back(opts.device_map[logical]);
  std::sort(requested.begin(), requested.end());
  requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
  const auto clocks = query_sm_clocks(requested);
  const int expected = opts.expected_clock_mhz > 0
                           ? opts.expected_clock_mhz
                           : (cal ? cal->sm_clock_mhz : 0);
  bool mismatch = false;
  if (expected > 0) {
    for (int clock : clocks) mismatch |= clock != expected;
  }
  if (mismatch) {
    const std::string message = "sampled SM clock does not equal expected lock " +
                                std::to_string(expected) + " MHz";
    if (opts.strict_clock) throw std::runtime_error(message);
    std::cerr << "WARNING: " << message << "; samples will retain clock values for filtering.\n";
  }
  if (cal && !cal->sm_clock_locked)
    std::cerr << "WARNING: calibration reports sm_clock_locked=false; using frequency-sampling fallback.\n";
}

std::pair<double, double> tier_values(Stage stage, const Calibration& cal,
                                      bool raw) {
  const TierCalibration& tier = stage == Stage::S2 ? cal.slow : cal.fast;
  return raw ? std::make_pair(tier.raw, tier.overhead_ns)
             : std::make_pair(tier.target, tier.overhead_ns);
}

void run_stage_once(const Options& opts, Stage stage, size_t data_size,
                    const Calibration& cal, const std::string& mode,
                    SampleWriter& samples, std::vector<CsvRow>& rows) {
  const auto logical = logical_pair(stage);
  if (!legal_pair(stage, logical[0], logical[1])) throw std::logic_error("illegal representative pair");
  PairContext ctx(logical, opts.device_map);
  const size_t transfer = transfer_size(stage, data_size);
  ctx.allocate(stage, transfer);
  const TierCalibration& tier = stage == Stage::S2 ? cal.slow : cal.fast;
  const PaceSpec pace{true, tier.target, tier.raw, tier.overhead_ns};
  const RunResult result = run_contexts({&ctx}, stage, transfer, opts.chunk,
                                         pace, opts.warmup, opts.iters,
                                         opts.sample_clocks);
  samples.add(mode, stage, data_size, {&ctx}, result);
  rows.push_back(make_row(mode, stage, data_size, transfer, opts.chunk,
                          tier.target, opts.ratio, ctx, result, 0));
  if (opts.verify) verify_result(ctx, stage, transfer, opts.chunk, pace);
}

void run_calibrate(const Options& opts) {
  if (!opts.skip_timer_check) {
    const double error = timer_self_test(opts.device_map[0]);
    std::cout << "%globaltimer self-test median relative error=" << error * 100 << "%\n";
    if (error > 0.01) throw std::runtime_error("%globaltimer self-test exceeded 1% tolerance");
  }
  enforce_clock(opts);
  SampleWriter samples(opts.out);
  std::vector<CsvRow> rows;
  Calibration cal;
  cal.chunk = opts.chunk;
  cal.fast = calibrate_tier(opts, Stage::S1, opts.fast_bw, samples, rows);
  cal.slow = calibrate_tier(opts, Stage::S2, opts.fast_bw / opts.ratio, samples, rows);
  std::vector<int> clock_devices{opts.device_map[0], opts.device_map[2], opts.device_map[4]};
  std::sort(clock_devices.begin(), clock_devices.end());
  clock_devices.erase(std::unique(clock_devices.begin(), clock_devices.end()), clock_devices.end());
  const auto clocks = query_sm_clocks(clock_devices);
  cal.sm_clock_mhz = opts.expected_clock_mhz > 0 ? opts.expected_clock_mhz : 0;
  cal.sm_clock_locked = opts.expected_clock_mhz > 0 &&
      std::all_of(clocks.begin(), clocks.end(), [&](int clock) {
        return clock == opts.expected_clock_mhz;
      });
  write_calibration(opts.calibration, cal);
  write_summary(opts.out, rows);
  std::cout << "Wrote " << opts.calibration << ", " << opts.out << ", and "
            << samples.path() << ".\n";
  auto endpoint_ok = [](const TierCalibration& tier) {
    return std::all_of(tier.endpoint_achieved.begin(), tier.endpoint_achieved.end(),
                       [&](double achieved) {
      return std::abs(achieved - tier.target) / tier.target <= 0.03;
    });
  };
  if (!endpoint_ok(cal.fast) || !endpoint_ok(cal.slow))
    std::cerr << "WARNING: endpoint cross-validation failed; continuing because "
                 "the sweep will export empirical per-message ASTRA-SIM profiles. "
                 "The single-bandwidth calibration must not be treated as valid "
                 "for the full message-size range.\n";
  if (std::any_of(rows.begin(), rows.end(), [](const CsvRow& row) { return row.iters < 100; }))
    throw std::runtime_error("calibration artifacts were written, but fewer than 100 samples survived frequency filtering");
}

void validate_calibration_compatibility(const Options& opts, const Calibration& cal) {
  if (cal.chunk != opts.chunk)
    throw std::runtime_error("--chunk differs from calibration.json; recalibrate or use matching chunk");
  const double fast_error = std::abs(cal.fast.target - opts.fast_bw) / opts.fast_bw;
  const double slow_target = opts.fast_bw / opts.ratio;
  const double slow_error = std::abs(cal.slow.target - slow_target) / slow_target;
  if (fast_error > 1e-6 || slow_error > 1e-6)
    throw std::runtime_error("target bandwidth/ratio differs from calibration.json");
  if (cal.fast.raw < 3 * cal.fast.target || cal.slow.raw < 3 * cal.slow.target)
    throw std::runtime_error("calibration violates B_raw >= 3*B_target");
  bool endpoint_failed = false;
  for (const TierCalibration* tier : {&cal.fast, &cal.slow}) {
    for (double achieved : tier->endpoint_achieved) {
      if (std::abs(achieved - tier->target) / tier->target > 0.03)
        endpoint_failed = true;
    }
  }
  if (endpoint_failed)
    std::cerr << "WARNING: calibration failed its 2/256 MiB endpoint "
                 "cross-validation; sweep results will be consumed as "
                 "per-message empirical ASTRA-SIM profiles.\n";
}

void run_stage_or_sweep(const Options& opts) {
  const Calibration cal = load_calibration(opts.calibration);
  validate_calibration_compatibility(opts, cal);
  enforce_clock(opts, &cal);
  if (!opts.skip_timer_check) {
    const double error = timer_self_test(opts.device_map[0]);
    if (error > 0.01) throw std::runtime_error("%globaltimer self-test exceeded 1% tolerance");
  }
  SampleWriter samples(opts.out);
  std::vector<CsvRow> rows;
  if (opts.mode == "stage") {
    run_stage_once(opts, opts.stage, opts.data_size, cal, "stage", samples, rows);
  } else {
    const std::array<size_t, 5> sizes{{2 * kMiB, 8 * kMiB, 32 * kMiB,
                                      64 * kMiB, 256 * kMiB}};
    for (size_t size : sizes)
      for (Stage stage : {Stage::S1, Stage::S2, Stage::S3})
        run_stage_once(opts, stage, size, cal, "sweep", samples, rows);
  }
  write_summary(opts.out, rows);
  std::cout << "Wrote " << opts.out << " and " << samples.path() << ".\n";
}

void run_concurrency(const Options& opts) {
  const Calibration cal = load_calibration(opts.calibration);
  validate_calibration_compatibility(opts, cal);
  enforce_clock(opts, &cal);
  SampleWriter samples(opts.out);
  std::vector<CsvRow> rows;
  const std::array<std::array<int, 2>, 4> y_pairs{{{0, 2}, {1, 3}, {4, 6}, {5, 7}}};
  for (int pair_count : {1, 2, 4}) {
    std::vector<std::unique_ptr<PairContext>> owners;
    std::vector<PairContext*> contexts;
    for (int p = 0; p < pair_count; ++p) {
      if (!legal_pair(Stage::S1, y_pairs[p][0], y_pairs[p][1]))
        throw std::logic_error("illegal concurrency pair");
      owners.push_back(std::make_unique<PairContext>(y_pairs[p], opts.device_map));
      owners.back()->allocate(Stage::S1, opts.data_size);
      contexts.push_back(owners.back().get());
    }
    const PaceSpec pace{true, cal.fast.target, cal.fast.raw, cal.fast.overhead_ns};
    const RunResult result = run_contexts(contexts, Stage::S1, opts.data_size,
                                           opts.chunk, pace, opts.warmup,
                                           opts.iters, opts.sample_clocks);
    const std::string mode = "concurrency-" + std::to_string(pair_count);
    samples.add(mode, Stage::S1, opts.data_size, contexts, result);
    for (size_t p = 0; p < contexts.size(); ++p)
      rows.push_back(make_row(mode, Stage::S1, opts.data_size, opts.data_size,
                              opts.chunk, cal.fast.target, opts.ratio,
                              *contexts[p], result, p));
    if (opts.verify)
      for (PairContext* ctx : contexts)
        verify_result(*ctx, Stage::S1, opts.data_size, opts.chunk, pace);
  }
  write_summary(opts.out, rows);
  std::cout << "Wrote " << opts.out << " and " << samples.path() << ".\n";
}

void assert_static_topology() {
  const std::array<std::array<int, 2>, 4> forbidden{{{0, 3}, {1, 2}, {4, 7}, {5, 6}}};
  for (Stage stage : {Stage::S1, Stage::S2, Stage::S3}) {
    const auto pair = logical_pair(stage);
    if (!legal_pair(stage, pair[0], pair[1])) throw std::logic_error("bad representative pair");
    for (const auto& invalid : forbidden) {
      if (legal_pair(stage, invalid[0], invalid[1]))
        throw std::logic_error("forbidden diagonal link admitted");
    }
  }
}

int main(int argc, char** argv) {
  try {
    const Options opts = parse_options(argc, argv);
    assert_static_topology();
    print_environment(opts);
    if (opts.mode == "calibrate") run_calibrate(opts);
    else if (opts.mode == "concurrency") run_concurrency(opts);
    else run_stage_or_sweep(opts);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
