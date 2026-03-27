# 第一周行动计划（Phase 0）

> 目标：搭好项目骨架，能读取并打印 GGUF 文件的全部元数据，理解模型文件内部结构。
> 截止状态：运行 `./build/tools/gguf_dump models/qwen2-0_5b-instruct-q4_k_m.gguf` 能打印出所有 key-value 元数据和 tensor 列表。

---

## Day 1（周一）：CMake 项目骨架

**目标**：项目能编译，跑通一个空的 Hello World，CI 配置就位。

### 任务清单

- [ ] 创建顶层 `CMakeLists.txt`，设置 C++17、Release/Debug 两个 preset
- [ ] 创建 `CMakePresets.json`，支持 `cmake --preset debug` 和 `cmake --preset release`
- [ ] 创建 `cmake/DetectSIMD.cmake`，检测当前机器是否支持 AVX2（用 `cpuid` 或 `check_cxx_source_runs`）
- [ ] 创建 `cmake/FetchDeps.cmake`，用 `FetchContent` 预留 GoogleTest 的引入位置（先注释掉，占坑）
- [ ] 在 `tools/gguf_dump/` 下创建 `main.cpp`，内容只有 `int main() { return 0; }`，确认能编译
- [ ] 创建 `.github/workflows/ci.yml`，触发条件：push 到 main，步骤：`cmake --preset release && cmake --build build`
- [ ] 创建 `.gitignore`（C++ 项目标准模板：`build/`、`*.o`、`*.a` 等）

### 验收

```bash
cmake --preset release
cmake --build build --target gguf_dump -j$(nproc)
./build/tools/gguf_dump/gguf_dump   # 正常退出，exit code 0
```

---

## Day 2（周二）：GGUF 格式阅读与数据结构定义

**目标**：不写解析代码，先把数据结构和接口设计好，做到"心里有数"再动手。

### 任务清单

- [ ] 通读 [GGUF 规范文档](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)，预计 2 小时，边读边记笔记
- [ ] 用 Python 写一个最简单的 GGUF 读取脚本（`scripts/inspect_gguf.py`），手动 `struct.unpack` 读 Magic、Version、n_tensors、n_kv，在终端打印出来，验证自己对格式的理解
- [ ] 在 `include/kestrel/loader/gguf_types.h` 中定义以下枚举（纯头文件，不需要 `.cpp`）：
  - `GGUFValueType`（uint8 ~ float64，共 13 种）
  - `GGMLType`（F32、F16、BF16、Q4_K、Q8_0 等，先列出常用的 10 种即可）
- [ ] 在 `include/kestrel/loader/gguf_parser.h` 中写好 `GGUFParser` 类的**声明**（不实现），列出所有 public 方法的签名和注释

### 验收

```bash
# Python 脚本能正确输出
python3 scripts/inspect_gguf.py models/qwen2-0_5b-instruct-q4_k_m.gguf
# 预期输出：
# Magic: GGUF
# Version: 3
# n_tensors: 290
# n_kv: 23
```

---

## Day 3（周三）：实现 GGUF Header 和 KV 元数据解析

**目标**：能读取并打印所有 key-value 元数据（不含 tensor 部分）。

### 任务清单

- [ ] 在 `src/loader/gguf_parser.cpp` 中实现文件打开（`mmap`）和 header 解析：
  - 验证 Magic number（`0x46554747`）
  - 读取 Version、n_tensors、n_kv
- [ ] 实现 `read_gguf_string()`（先读 uint64 长度，再读字符数据）
- [ ] 实现 `read_value(GGUFValueType)`，处理以下类型：
  - 标量：uint8/int8/uint16/int16/uint32/int32/float32/bool/uint64/int64/float64
  - string：调用 `read_gguf_string()`
  - array：递归读取元素（每个元素是相同类型的 value）
- [ ] 实现 `parse_metadata()`，将所有 kv 对存入 `std::unordered_map<std::string, GGUFValue>`
- [ ] 在 `tools/gguf_dump/main.cpp` 中调用以上接口，遍历打印所有元数据

### 重点注意

- `mmap` 在 Linux 上用 `open` + `mmap(PROT_READ, MAP_PRIVATE)`，读完记得 `munmap`
- 所有读取都用 `memcpy` 而非直接指针解引用（避免未对齐访问 UB）
- array 类型的元素个数是 uint64，不是 uint32

### 验收

```bash
./build/tools/gguf_dump/gguf_dump models/qwen2-0_5b-instruct-q4_k_m.gguf
# 预期输出（部分）：
# [metadata] general.architecture = "qwen2"
# [metadata] general.name = "Qwen2-0.5B-Instruct"
# [metadata] qwen2.block_count = 24
# [metadata] qwen2.embedding_length = 896
# [metadata] qwen2.attention.head_count = 14
# [metadata] qwen2.attention.head_count_kv = 2
# [metadata] tokenizer.ggml.model = "gpt2"
# ... (共 23 条)
```

---

## Day 4（周四）：实现 Tensor Info 解析

**目标**：能打印所有 tensor 的名字、形状、数据类型、在文件中的偏移量。

### 任务清单

- [ ] 定义 `TensorInfo` 结构体（`name`、`shape`、`dtype`、`offset`）
- [ ] 实现 `parse_tensor_infos()`：
  - 读取 name（gguf string 格式）
  - 读取 n_dims（uint32），然后读取 n_dims 个 uint64 维度值
  - 读取 dtype（uint32，映射到 `GGMLType`）
  - 读取 offset（uint64，相对于数据区起始的字节偏移）
- [ ] 计算数据区起始位置：所有 tensor info 读完后，当前指针对齐到 32 字节边界（`GGUF_DEFAULT_ALIGNMENT = 32`）
- [ ] 在 dump 工具中补充打印 tensor 列表，格式：`[tensor] name | shape | dtype | offset`
- [ ] 统计并打印总体信息：tensor 总数、数据区总大小

### 验收

```bash
./build/tools/gguf_dump/gguf_dump models/qwen2-0_5b-instruct-q4_k_m.gguf
# 预期输出（部分）：
# [tensor]  token_embd.weight          | [1024, 896]   | Q4_K  | offset=0
# [tensor]  blk.0.attn_norm.weight     | [896]         | F32   | offset=...
# [tensor]  blk.0.attn_q.weight        | [896, 896]    | Q4_K  | offset=...
# ...
# Total tensors: 290
# Data section size: 394.2 MB
```

---

## Day 5（周五）：提取 ModelConfig + 写第一个单元测试

**目标**：从元数据中提取结构化的模型配置，并写第一个可运行的测试。

### 任务清单

- [ ] 在 `include/kestrel/types.h` 中定义 `ModelConfig` 结构体：
  ```cpp
  struct ModelConfig {
      std::string arch;         // "qwen2" / "llama"
      int32_t n_layers;
      int32_t n_heads;
      int32_t n_kv_heads;
      int32_t hidden_dim;
      int32_t ffn_hidden_dim;
      int32_t vocab_size;
      int32_t max_seq_len;
      float   rope_theta;
  };
  ```
- [ ] 在 `GGUFParser` 中实现 `ModelConfig extract_config()` 方法，从 metadata map 中读取对应字段（注意：Qwen2 的 key 前缀是 `qwen2.*`，Llama 是 `llama.*`）
- [ ] 引入 GoogleTest（在 `cmake/FetchDeps.cmake` 中用 `FetchContent_Declare` + `FetchContent_MakeAvailable`）
- [ ] 在 `tests/unit/test_gguf_parser.cpp` 中写第一个测试：
  - 加载 `qwen2-0_5b-instruct-q4_k_m.gguf`
  - 断言 `config.arch == "qwen2"`
  - 断言 `config.n_layers == 24`
  - 断言 `config.n_kv_heads == 2`
  - 断言 tensor 总数 == 290
- [ ] 配置 `tests/CMakeLists.txt`，能用 `ctest` 运行测试

### 验收

```bash
cmake --build build --target test_gguf_parser
cd build && ctest --output-on-failure
# [----------] 1 test from GGUFParserTest
# [ RUN      ] GGUFParserTest.LoadQwen2Config
# [       OK ] GGUFParserTest.LoadQwen2Config (123 ms)
# [  PASSED  ] 1 test.
```

---

## Day 6-7（周末）：整理与缓冲

**目标**：消化本周内容，弥补未完成项，写周报笔记。

### 任务清单

- [ ] 回顾 Day 1-5，把所有未完成的 `[ ]` 补完
- [ ] 写 `docs/notes/week1.md`：记录本周遇到的问题、对 GGUF 格式的理解、和规范文档中不一致的地方
- [ ] 用 Python 对照验证：写 `scripts/compare_with_gguf_lib.py`，用 `pip install gguf` 官方库读取同一个文件，逐字段对比你的解析结果是否一致
- [ ] 阅读 llama.cpp 的 `ggml/src/ggml.h` 前 200 行，理解 `ggml_tensor` 结构体设计，和你的 TensorInfo 做对比，记录设计差异

### 验收

```bash
python3 scripts/compare_with_gguf_lib.py models/qwen2-0_5b-instruct-q4_k_m.gguf
# All metadata matched: 23/23 ✓
# All tensor infos matched: 290/290 ✓
```

---

## 本周结束状态检查

完成本周后，你应该能回答以下问题（不看资料）：

- GGUF 文件由哪四个部分组成？各部分的顺序是什么？
- `n_kv` 和 `n_tensors` 各自存在文件的哪个位置？用什么类型存储？
- GGUF string 是怎么编码的？为什么不用 null-terminated string？
- tensor 的 `offset` 是相对于什么的偏移？数据区起始是怎么对齐的？
- Qwen2-0.5B 有多少层？每层有多少个 attention head？KV head 是多少？
- Q4_K 格式的 tensor 和 F32 格式的 tensor，在 tensor info 中有什么不同？

---

## 参考资料

| 资料 | 用途 | 预计阅读时间 |
|------|------|-------------|
| [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) | Day 2 必读，格式权威定义 | 2h |
| [llama.cpp gguf.h](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py) | Python 参考实现，用于对照验证 | 1h |
| [ggml.h 前200行](https://github.com/ggerganov/ggml/blob/master/include/ggml.h) | 理解 ggml_tensor 设计 | 0.5h |
| `mmap(2)` man page | Day 3 实现文件读取前查阅 | 0.5h |

---

## 下周预告（Day 8-14）：Tensor 类和第一个算子

完成本周后，下周将开始实现 `Tensor` 类（shape、dtype、data_ptr、strides）和 `MemoryArena` 内存分配器，并实现第一个算子：标量版本的矩阵乘法，配套数值精度测试。
