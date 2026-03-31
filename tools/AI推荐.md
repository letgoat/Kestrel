# AI 方向职业规划推荐

> **个人背景**：985本科 · C++/Python · 驱动工程师1年 · 鸿蒙框架应用开发0.5年 · 工作年限1.5年
> **技能储备**：数据库 · 网络编程 · Docker · Redis · 操作系统

---

## 一、AI 方向薪资对比（不同年限）

> 数据来源于国内一线/准一线城市（北京/上海/深圳/杭州），以2024-2025年市场行情为参考。

| AI 方向 | 1-3年（万/年） | 3-5年（万/年） | 5-8年（万/年） | 8年+（万/年） | 与你背景匹配度 |
|---|---|---|---|---|---|
| **AI Infra / MLSys（推理/训练系统）** | 40-70 | 70-120 | 120-200 | 200-400+ | ⭐⭐⭐⭐⭐ |
| **大模型推理优化（LLM Inference）** | 40-70 | 80-130 | 130-220 | 200-400+ | ⭐⭐⭐⭐⭐ |
| **算法工程师（NLP/CV/多模态）** | 35-60 | 60-100 | 100-160 | 160-300 | ⭐⭐⭐ |
| **AI 应用/大模型应用开发** | 25-45 | 40-70 | 60-100 | 100-150 | ⭐⭐⭐ |
| **自动驾驶（感知/规控/系统）** | 35-65 | 65-110 | 110-180 | 180-300+ | ⭐⭐⭐⭐ |
| **AI芯片/编译器（CUDA/NPU）** | 45-80 | 80-150 | 150-250 | 250-500+ | ⭐⭐⭐⭐ |

### 综合评估

结合你的背景（C++底层经验、驱动开发、操作系统、网络编程），**最推荐以下两个方向**：

1. **AI Infra / MLSys — 模型训练与推理系统**
2. **大模型推理优化（LLM Inference Engineering）**

这两个方向**强依赖 C++ 底层能力 + 系统知识**，是你现有背景的最佳延伸，且薪资天花板极高，头部公司（字节、阿里、腾讯、百度、华为、各大 AI 创业公司）需求旺盛。

---

## 二、重点推荐方向详解

---

### 方向一：AI Infra / MLSys（机器学习系统）

#### 核心定位

负责 AI 训练/推理基础设施的设计与优化，包括：分布式训练框架、显存管理、算子优化、模型并行等。直接决定大模型训练效率，是 AI 公司最核心的底层岗位之一。

#### 需要掌握的技术栈

**基础层（必须）**

| 技术 | 说明 |
|---|---|
| C++17/20 | 高性能算子与框架实现 |
| CUDA 编程 | GPU 并行计算、内存管理、Kernel 优化 |
| Python | 框架 API 层、训练脚本 |
| 数据结构与算法 | 图结构、内存池、调度算法 |
| 操作系统 | 进程/线程/内存管理（你已有基础） |
| 计算机体系结构 | CPU/GPU 架构、缓存层次、SIMD |

**AI 系统层（核心）**

| 技术 | 说明 |
|---|---|
| PyTorch 源码 | Autograd、Dispatcher、TorchScript |
| 分布式训练 | Data Parallel / Tensor Parallel / Pipeline Parallel |
| 通信库 | NCCL、MPI、RDMA、InfiniBand |
| 显存优化 | Activation Checkpointing、混合精度 FP16/BF16 |
| 算子融合 | Kernel Fusion、Flash Attention 原理 |
| 模型量化 | INT8/INT4、GPTQ、AWQ |

**工程层（加分）**

| 技术 | 说明 |
|---|---|
| Docker / Kubernetes | 训练集群管理（你已有基础） |
| RDMA / 网络编程 | 高速互联（你已有基础） |
| Profiling 工具 | Nsight、PyTorch Profiler、perf |

#### 学习资源

**书籍**

- 《CUDA C++ Programming Guide》— NVIDIA 官方文档，必读
- 《深入理解计算机系统（CSAPP）》— 系统基础（你可快速过）
- 《Dive into Deep Learning（d2l.ai）》— 深度学习基础
- 《Designing Machine Learning Systems》— MLSys 系统视角

**课程**

| 课程 | 链接 | 说明 |
|---|---|---|
| CMU 15-418 并行计算 | https://www.cs.cmu.edu/~418/ | GPU/并行编程必学 |
| Stanford CS149 | https://cs149.stanford.edu | 并行计算进阶 |
| MIT 6.S965 TinyML | https://efficientml.ai | 模型压缩与推理优化 |
| CUDA官方教程 | https://developer.nvidia.com/cuda-education | 入门到进阶 |
| 台大李宏毅机器学习 | https://speech.ee.ntu.edu.tw/~hylee/ml/ | 深度学习理论基础 |

**博客 / 文章**

- Lilian Weng 博客：https://lilianweng.github.io
- HuggingFace 博客：https://huggingface.co/blog
- Towards Data Science：https://towardsdatascience.com
- 知乎专栏「深度学习系统」

#### 推荐开源项目

| 项目 | 仓库地址 | 学习重点 |
|---|---|---|
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | C++推理、量化、GGUF格式，入门首选 |
| **vLLM** | https://github.com/vllm-project/vllm | PagedAttention、高吞吐推理调度 |
| **FlashAttention** | https://github.com/Dao-AILab/flash-attention | CUDA Kernel优化经典实现 |
| **PyTorch** | https://github.com/pytorch/pytorch | Autograd/Dispatcher源码精读 |
| **Megatron-LM** | https://github.com/NVIDIA/Megatron-LM | 分布式训练框架 |
| **DeepSpeed** | https://github.com/microsoft/DeepSpeed | ZeRO优化、显存管理 |
| **TVM** | https://github.com/apache/tvm | 编译器优化、算子自动调优 |
| **CUTLASS** | https://github.com/NVIDIA/cutlass | CUDA矩阵运算模板库 |

#### 学习路径建议

```
第1-2个月：深度学习基础 + CUDA入门
    └── d2l.ai 前8章 + CUDA编程基础 + GPU架构理解

第3-4个月：PyTorch 内核 + llama.cpp 精读
    └── PyTorch Autograd源码 + llama.cpp完整阅读并尝试修改

第5-6个月：推理优化专项
    └── vLLM源码 + FlashAttention论文+代码 + 量化技术实践

第7-9个月：分布式训练
    └── Megatron-LM + DeepSpeed + NCCL通信原理

第10-12个月：完整项目 + 刷题 + 面试
    └── 做一个端到端的推理优化项目，写博客，准备面试
```

---

### 方向二：大模型推理优化（LLM Inference Engineering）

#### 核心定位

专注于大语言模型（LLM）在生产环境中的部署与推理加速，包括：KV Cache 优化、量化、投机解码、推理服务化、边缘端部署等。是目前需求最旺盛、增长最快的 AI 岗位之一。

#### 需要掌握的技术栈

**基础层（必须）**

| 技术 | 说明 |
|---|---|
| C++ / Python | 推理引擎核心用 C++，服务层用 Python |
| CUDA / GPU 架构 | Tensor Core、warp、shared memory |
| Transformer 架构 | Attention机制、位置编码、KV Cache |
| 量化理论 | FP16、INT8、INT4、GPTQ、AWQ、SmoothQuant |

**推理系统层（核心）**

| 技术 | 说明 |
|---|---|
| KV Cache 管理 | PagedAttention、Dynamic batching |
| 投机解码 | Speculative Decoding、Draft Model |
| 连续批处理 | Continuous Batching、请求调度 |
| 模型并行 | Tensor/Pipeline Parallel 推理 |
| 算子优化 | Fused Kernel、GEMM优化 |

**部署层（加分）**

| 技术 | 说明 |
|---|---|
| Triton Inference Server | NVIDIA 推理服务框架 |
| gRPC / REST API | 推理服务接口（你已有网络编程基础） |
| Docker / K8s | 推理服务部署（你已有基础） |
| 边缘部署 | llama.cpp、MNN、NCNN（你有嵌入式/驱动背景） |

#### 学习资源

**必读论文**

| 论文 | 说明 |
|---|---|
| Attention Is All You Need (2017) | Transformer 基础 |
| FlashAttention (2022/2023) | IO-aware 注意力优化 |
| Efficient Memory Management for LLM (vLLM, 2023) | PagedAttention |
| LLM.int8() / GPTQ / AWQ | 量化系列论文 |
| Speculative Decoding (2023) | 投机解码 |
| Continuous Batching (Orca, 2022) | 动态批处理 |

**书籍 / 文档**

- 《大模型推理优化实践》— 知乎/博客系列文章
- NVIDIA TensorRT 官方文档：https://docs.nvidia.com/deeplearning/tensorrt/
- vLLM 官方文档：https://docs.vllm.ai

**课程**

| 课程 | 链接 | 说明 |
|---|---|---|
| MIT 6.S965 高效深度学习 | https://efficientml.ai | 量化/剪枝/推理优化系统课 |
| Fast.ai | https://course.fast.ai | 深度学习实践 |
| Andrej Karpathy: Neural Networks Zero to Hero | https://karpathy.ai/zero-to-hero.html | LLM原理从零实现 |

**博客推荐**

- https://huggingface.co/blog/llm-inference — HuggingFace推理优化系列
- https://www.databricks.com/blog — LLM服务化实践
- https://zhuanlan.zhihu.com/p/638468472 — 国内LLM推理优化总结

#### 推荐开源项目

| 项目 | 仓库地址 | 学习重点 |
|---|---|---|
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | 最佳入门项目，纯C++推理 |
| **vLLM** | https://github.com/vllm-project/vllm | 生产级推理引擎，重点学习调度 |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA官方LLM推理优化 |
| **MLC-LLM** | https://github.com/mlc-ai/mlc-llm | 跨平台编译部署 |
| **SGLang** | https://github.com/sgl-project/sglang | 高性能推理框架，RadixAttention |
| **Ollama** | https://github.com/ollama/ollama | 本地部署生态，Go+llama.cpp |
| **nanoGPT** | https://github.com/karpathy/nanoGPT | 从零实现GPT，理解原理 |
| **ExLlamaV2** | https://github.com/turboderp/exllamav2 | 高效量化推理 |

#### 学习路径建议

```
第1个月：Transformer 原理 + nanoGPT 实现
    └── 读《Attention Is All You Need》+ 跟 Karpathy 视频从零实现 GPT

第2-3个月：llama.cpp 深度学习
    └── 阅读全部源码 → 理解 GGUF 格式 → 理解量化实现 → 尝试添加新功能

第4-5个月：GPU推理优化
    └── FlashAttention 论文+代码 → CUDA Kernel 写注意力算子 → Triton 语言

第6-7个月：vLLM 源码精读
    └── PagedAttention → Continuous Batching → 调度器逻辑

第8-9个月：生产部署实践
    └── TensorRT-LLM 实践 → 量化全流程（GPTQ/AWQ）→ 服务化部署

第10-12个月：项目 + 面试
    └── 完整的推理服务项目（含量化+批处理+API） → 面试准备
```

---

## 三、求职目标公司参考

| 公司 | 方向 | 备注 |
|---|---|---|
| 字节跳动 | AI Infra、LLM推理 | 豆包大模型团队，薪资顶级 |
| 阿里云/通义 | MLSys、推理优化 | 规模大，岗位多 |
| 腾讯混元 | AI Infra | 待遇好 |
| 华为 | AI芯片、推理框架 | MindSpore、昇腾，C++强 |
| 百度飞桨 | 训练/推理框架 | PaddlePaddle生态 |
| 月之暗面/Kimi | LLM推理 | 创业公司，期权空间大 |
| DeepSeek | AI Infra | 技术氛围好，有影响力 |
| 智谱AI / MiniMax | MLSys | 快速成长期，机会多 |
| 小红书/快手 | 推理基础设施 | 业务驱动，落地场景丰富 |

---

## 四、过渡建议

鉴于你目前是驱动/框架背景，建议按以下步骤过渡：

1. **近期（0-3个月）**：补齐深度学习基础，读懂 Transformer，运行并阅读 `llama.cpp` 源码
2. **中期（3-9个月）**：攻克 CUDA 编程，精读 vLLM / FlashAttention，做出完整项目
3. **求职期（9-12个月）**：刷 LeetCode（100题以上）+ 系统设计准备 + 投递相关岗位

> **关键优势**：你的 C++ 底层能力 + 操作系统 + 驱动背景，在 AI Infra 岗位中是**稀缺竞争力**，很多纯算法出身的候选人反而缺少这些。充分利用这一差异化优势。

---

*生成日期：2026-03-31*
