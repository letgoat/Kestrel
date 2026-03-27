# 第二周行动计划（Phase 1 开始）

> 目标：实现 `Tensor` 类和 `MemoryArena` 内存分配器，完成标量版矩阵乘法、RMSNorm、Softmax 三个算子，每个算子都有数值精度测试（对比 NumPy/PyTorch 参考输出）。
> 截止状态：运行 `ctest` 全部通过，`bench_matmul` 能输出标量版本的 GFLOPS 数字。

---

## Day 8（周一）：DType 系统 + MemoryArena

**目标**：搞定内存管理基础设施，后续所有 Tensor 分配都依赖它。

### 任务清单

- [ ] 在 `include/kestrel/types.h` 中定义 `DType` 枚举和辅助函数：
  ```cpp
  enum class DType : uint8_t { F32, F16, BF16, I32, I8, Q4_K, Q8_0 };

  size_t dtype_size(DType dt);        // F32→4, F16→2, I8→1, Q4_K→特殊处理
  const char* dtype_name(DType dt);   // 用于打印调试
  DType ggml_type_to_dtype(uint32_t ggml_type);  // 从 GGMLType 转换
  ```
- [ ] 在 `src/tensor/memory_arena.cpp` 中实现 `MemoryArena`：
  - 构造函数：`MemoryArena(size_t capacity, size_t alignment = 64)`，用 `posix_memalign` 分配
  - `void* allocate(size_t bytes)`：bump pointer 分配，O(1)，超出容量抛 `std::bad_alloc`
  - `void reset()`：将 `used_` 归零（不 free 内存，整体复用）
  - `size_t used() const` 和 `size_t capacity() const`
- [ ] 在 `tests/unit/test_memory_arena.cpp` 中写测试：
  - 分配多个大小不同的块，验证每次返回的指针都是 64 字节对齐
  - 验证 `reset()` 后再次分配返回相同基地址
  - 验证超出容量时抛出异常

### 重点注意

- `posix_memalign` 第二个参数（对齐值）必须是 2 的幂且 >= `sizeof(void*)`，64 满足
- bump pointer 对齐：`size_t aligned = (bytes + alignment_ - 1) & ~(alignment_ - 1);`
- 析构函数里调用 `free(base_)`，不能用 `delete`（因为是 `posix_memalign` 分配的）

### 验收

```bash
cmake --build build --target test_memory_arena && ctest -R test_memory_arena -V
# [ RUN      ] MemoryArenaTest.AlignmentGuarantee   → OK
# [ RUN      ] MemoryArenaTest.ResetReusesMemory     → OK
# [ RUN      ] MemoryArenaTest.ThrowsOnOverflow       → OK
```

---

## Day 9（周二）：Tensor 类实现

**目标**：Tensor 类能表达形状、步幅、dtype，支持基础的视图操作（不拷贝数据）。

### 任务清单

- [ ] 在 `include/kestrel/tensor.h` 中定义 `Tensor` 类，核心字段：
  ```cpp
  class Tensor {
  public:
      std::vector<int64_t> shape;
      std::vector<int64_t> strides;  // 行优先，单位：元素个数（不是字节）
      DType   dtype;
      void*   data_ptr;   // 裸指针，生命周期由外部 Arena 管理

      static Tensor empty(std::vector<int64_t> shape, DType dtype,
                          MemoryArena& arena);
      static Tensor from_raw(void* ptr, std::vector<int64_t> shape, DType dtype);

      int64_t numel() const;   // 元素总数
      int64_t nbytes() const;  // 字节总数
      int     ndim() const;

      // 视图操作（不拷贝，返回新 Tensor 共享同一 data_ptr）
      Tensor view(std::vector<int64_t> new_shape) const;
      Tensor slice(int dim, int64_t start, int64_t end) const;

      // 访问元素（仅用于测试和调试，不用于性能路径）
      float  at_f32(std::initializer_list<int64_t> indices) const;
      float* data_f32() const { return static_cast<float*>(data_ptr); }
  };
  ```
- [ ] 实现行优先步幅计算：`strides[ndim-1] = 1`，`strides[i] = strides[i+1] * shape[i+1]`
- [ ] 实现 `view()`：检查 numel 相同，重新计算步幅
- [ ] 实现 `slice(dim, start, end)`：修改对应维度的 shape 和 data_ptr 偏移
- [ ] 在 `tests/unit/test_tensor.cpp` 中写测试：
  - 创建 `[4, 8]` F32 Tensor，验证 `numel()==32`、`nbytes()==128`
  - 验证步幅：`strides == {8, 1}`
  - 验证 `slice(0, 1, 3)` 返回 shape `[2, 8]` 且 data_ptr 偏移正确
  - 验证 `view({32})` 返回 shape `[32]`，步幅 `{1}`

### 验收

```bash
cmake --build build --target test_tensor && ctest -R test_tensor -V
# 全部通过
```

---

## Day 10（周三）：生成 PyTorch 参考数据 + 矩阵乘法（标量版）

**目标**：先用 Python 生成算子的参考输入输出，再实现 C++ 算子并对齐数值。

### 任务清单

**上午：生成参考数据（Python，约 1.5 小时）**

- [ ] 创建 `scripts/generate_test_data.py`，使用 PyTorch 生成以下数据并保存为 `.npy`：
  ```python
  import torch, numpy as np, os
  os.makedirs("tests/fixtures/ops", exist_ok=True)
  torch.manual_seed(42)

  # matmul: [32, 64] x [64, 128] = [32, 128]
  A = torch.randn(32, 64, dtype=torch.float32)
  B = torch.randn(64, 128, dtype=torch.float32)
  C = torch.matmul(A, B)
  np.save("tests/fixtures/ops/matmul_A.npy", A.numpy())
  np.save("tests/fixtures/ops/matmul_B.npy", B.numpy())
  np.save("tests/fixtures/ops/matmul_C_ref.npy", C.numpy())
  ```
- [ ] 运行脚本，确认文件生成：`python3 scripts/generate_test_data.py`

**下午：实现矩阵乘法（约 3 小时）**

- [ ] 在 `src/ops/matmul/matmul_scalar.cpp` 中实现朴素三重循环矩阵乘法：
  ```cpp
  // C[M×N] = A[M×K] × B[K×N]
  void matmul_f32_scalar(const float* A, const float* B, float* C,
                         int M, int K, int N) {
      for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) {
              float sum = 0.0f;
              for (int k = 0; k < K; ++k)
                  sum += A[m*K + k] * B[k*N + n];
              C[m*N + n] = sum;
          }
  }
  ```
- [ ] 在 `src/ops/matmul/matmul.h` 中声明统一接口 `void matmul_f32(...)`，内部调用标量版
- [ ] 在 `tests/unit/test_matmul.cpp` 中写精度测试：
  - 加载 `.npy` 参考数据（用自己写的极简 npy 读取函数，或内联 50 行实现）
  - 调用 `matmul_f32_scalar`
  - 逐元素比较，断言 `max_abs_error < 1e-5`

### 极简 .npy 读取（直接复制到测试辅助文件中）

```cpp
// tests/unit/test_helpers.h
#include <fstream>
// 只支持 float32，形状不超过 4 维，够用了
std::vector<float> load_npy_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    // 跳过 header（magic(6) + version(2) + header_len(2 or 4)）
    char magic[6]; f.read(magic, 6);
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint16_t hlen; f.read((char*)&hlen, 2);
    std::string header(hlen, ' '); f.read(&header[0], hlen);
    // 直接读剩余数据
    auto start = f.tellg();
    f.seekg(0, std::ios::end);
    size_t data_bytes = (size_t)f.tellg() - (size_t)start;
    f.seekg(start);
    std::vector<float> data(data_bytes / sizeof(float));
    f.read((char*)data.data(), data_bytes);
    return data;
}
```

### 验收

```bash
cmake --build build --target test_matmul && ctest -R test_matmul -V
# [ RUN      ] MatMulTest.ScalarF32Accuracy
# max_abs_error = 3.8e-6   atol = 1e-5   PASS
# [       OK ] MatMulTest.ScalarF32Accuracy
```

---

## Day 11（周四）：RMSNorm 算子

**目标**：实现 RMSNorm（Llama/Qwen 系列的归一化层），数值对齐 PyTorch。

### 背景知识（先理解再实现）

RMSNorm 公式：

```
rms  = sqrt( mean(x²) + ε )
out  = (x / rms) * weight
```

与 LayerNorm 的区别：不减均值（no mean subtraction），只做 RMS 缩放，计算量更少。

### 任务清单

- [ ] 在 `scripts/generate_test_data.py` 中追加生成 RMSNorm 参考数据：
  ```python
  # RMSNorm: input [8, 896], weight [896]
  x = torch.randn(8, 896)
  w = torch.randn(896)
  eps = 1e-5
  rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
  out = (x / rms) * w
  np.save("tests/fixtures/ops/rmsnorm_x.npy", x.numpy())
  np.save("tests/fixtures/ops/rmsnorm_w.npy", w.numpy())
  np.save("tests/fixtures/ops/rmsnorm_out_ref.npy", out.numpy())
  ```
- [ ] 在 `src/ops/layernorm.cpp` 中实现：
  ```cpp
  // x: [batch, dim]，weight: [dim]，out: [batch, dim]
  void rms_norm_f32(const float* x, const float* weight, float* out,
                    int batch, int dim, float eps = 1e-5f) {
      for (int b = 0; b < batch; ++b) {
          const float* xb = x + b * dim;
          float* ob = out + b * dim;
          // 计算 mean(x²)
          float sum_sq = 0.0f;
          for (int i = 0; i < dim; ++i) sum_sq += xb[i] * xb[i];
          float rms_inv = 1.0f / sqrtf(sum_sq / dim + eps);
          // 应用 weight
          for (int i = 0; i < dim; ++i)
              ob[i] = xb[i] * rms_inv * weight[i];
      }
  }
  ```
- [ ] 在 `tests/unit/test_ops.cpp` 中写 RMSNorm 精度测试（atol = 1e-5）

### 验收

```bash
ctest -R test_ops -V
# [ RUN      ] OpsTest.RMSNormAccuracy   → max_err < 1e-5   PASS
```

---

## Day 12（周五）：Softmax 算子 + 第一个 Benchmark

**目标**：完成 Softmax 实现，并跑出矩阵乘法的第一个性能数字（标量版基线）。

### 任务清单

**上午：Softmax（约 2 小时）**

- [ ] 生成参考数据（追加到 `generate_test_data.py`）：
  ```python
  # Softmax: [32, 1024]，沿最后一维
  logits = torch.randn(32, 1024)
  out = torch.softmax(logits, dim=-1)
  np.save("tests/fixtures/ops/softmax_in.npy", logits.numpy())
  np.save("tests/fixtures/ops/softmax_out_ref.npy", out.numpy())
  ```
- [ ] 在 `src/ops/activation.cpp` 中实现数值稳定的 Softmax：
  ```cpp
  void softmax_f32(const float* x, float* out, int batch, int dim) {
      for (int b = 0; b < batch; ++b) {
          const float* xb = x + b*dim;
          float* ob = out + b*dim;
          // 找最大值（数值稳定性）
          float max_val = *std::max_element(xb, xb + dim);
          float sum = 0.0f;
          for (int i = 0; i < dim; ++i) {
              ob[i] = expf(xb[i] - max_val);
              sum += ob[i];
          }
          float inv_sum = 1.0f / sum;
          for (int i = 0; i < dim; ++i) ob[i] *= inv_sum;
      }
  }
  ```
- [ ] 写精度测试（atol = 1e-6）

**下午：第一个 Benchmark（约 2 小时）**

- [ ] 引入 Google Benchmark（`cmake/FetchDeps.cmake` 中追加 `FetchContent`）
- [ ] 在 `benchmarks/bench_matmul.cpp` 中写基准测试：
  ```cpp
  #include <benchmark/benchmark.h>
  static void BM_MatMul_Scalar(benchmark::State& state) {
      int M = state.range(0), K = 4096, N = 4096;
      std::vector<float> A(M*K), B(K*N), C(M*N);
      // 初始化随机数据...
      for (auto _ : state) {
          matmul_f32_scalar(A.data(), B.data(), C.data(), M, K, N);
          benchmark::DoNotOptimize(C.data());
      }
      // 报告 GFLOPS
      double flops = 2.0 * M * K * N;
      state.counters["GFLOPS"] = benchmark::Counter(
          flops * state.iterations(), benchmark::Counter::kIsRate, 1e9);
  }
  BENCHMARK(BM_MatMul_Scalar)->Arg(1)->Arg(8)->Arg(32);
  BENCHMARK_MAIN();
  ```
- [ ] 编译运行，记录 M=1（decode 场景）和 M=32（prefill 场景）的 GFLOPS 数字到 `docs/notes/week2.md`

### 验收

```bash
./build/benchmarks/bench_matmul --benchmark_format=console
# BM_MatMul_Scalar/1    ...  GFLOPS=X.XX
# BM_MatMul_Scalar/32   ...  GFLOPS=X.XX
# （数字不重要，能跑出来就行，下周 AVX2 优化后对比用）
```

---

## Day 13-14（周末）：整理与自我测试

**目标**：全面验收本周产出，补漏，建立 PyTorch 交叉验证脚本。

### 任务清单

- [ ] 运行 `ctest --output-on-failure`，确保全部测试通过（本周累计应有 6+ 个测试）
- [ ] 写 `scripts/verify_ops.py`：用 PyTorch 实现同样的算子，加载 C++ 输出结果（通过写 `.npy` 文件），自动逐元素比较，输出 pass/fail 报告
- [ ] 写 `docs/notes/week2.md`：
  - 记录矩阵乘法标量版的性能数字（M=1 和 M=32 的 GFLOPS）
  - 记录 Softmax 数值稳定性问题（为什么要减 max_val，不减会怎样，实验验证）
  - 记录 RMSNorm 和 LayerNorm 的区别（用自己的话写，不要抄）
- [ ] 阅读 `ggml/src/ggml-cpu/ggml-cpu-impl.h` 中 `ggml_vec_dot_f32` 函数（约 30 行），和你的 matmul 内循环对比，记录差异
- [ ] 检查所有头文件是否有 `#pragma once` 保护

### 验收

```bash
python3 scripts/verify_ops.py
# matmul   scalar: max_err=3.8e-06  PASS ✓
# rms_norm        : max_err=8.1e-07  PASS ✓
# softmax         : max_err=2.3e-07  PASS ✓

ctest --output-on-failure
# 100% tests passed, 0 tests failed
```

---

## 本周结束状态检查

完成本周后，你应该能回答以下问题（不看资料）：

- bump pointer allocator 和 malloc 相比，有什么优势和限制？什么场景适合用它？
- Tensor 的 stride 是怎么计算的？`slice(dim=0, 0, 2)` 之后 stride 变了吗？
- 为什么 Softmax 实现时要先减去最大值？不减会发生什么（数值溢出还是精度问题）？
- RMSNorm 和 LayerNorm 的公式区别是什么？为什么 Llama 系列选 RMSNorm？
- 矩阵乘法标量版在你的机器上，M=1 时大约多少 GFLOPS？理论峰值是多少？差距在哪里？
- 什么是 atol（绝对误差容限）？为什么 F32 matmul 的 atol 设 1e-5 而不是 0？

---

## 参考资料

| 资料 | 用途 | 预计时间 |
|------|------|---------|
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 理解 RMSNorm 的论文（3页，很短） | 30min |
| [ggml_vec_dot_f32 源码](https://github.com/ggerganov/ggml/blob/master/src/ggml-cpu/ggml-cpu-impl.h) | 对比你的 matmul 内循环实现 | 30min |
| NumPy .npy 格式说明 | 理解 `load_npy_f32` 辅助函数的跳过逻辑 | 15min |
| Google Benchmark 官方文档 Getting Started | 配置 benchmark 时查阅 | 30min |

---

## 下周预告（Day 15-21）：AVX2 加速矩阵乘法 + RoPE 算子

完成本周后，下周将在 Day 10 的标量版矩阵乘法基础上，实现 AVX2 SIMD 加速版本，目标提升 5x+ 性能；同时实现 RoPE（Rotary Position Embedding），为 Week 4 的完整 Transformer forward pass 做准备。
