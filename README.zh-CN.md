# Ember

为 **NVIDIA DGX Spark（GB10，SM121）** 手写的 **Qwen3.8 Flash Next** 推理引擎，提供 OpenAI 兼容的 HTTP/SSE 接口，支持工具调用和思考预算。

[English](README.md)

> **状态：0.1.0，实验性。** 这是只针对一个模型、一种 GPU 的研究版本。下列数据在一台 DGX Spark 上测得；完整质量评测和四路 262K 上下文容量测试尚未在这个发布包上重跑。

## 主要技术

| 方向 | 内容 |
| --- | --- |
| 投机解码 | 原生 MTP 头 + DFlash2 块草稿器（NVFP4 权重、融合 Triton 内核），成本模型在两者间选择，单路 8 行验证 |
| MoE | SM121 W4A4 路由专家 MoE（NVFP4 激活 × NVFP4 权重，CuTe DSL），确定性 tactic 表，router/indexer 逐行 GEMV 合并 |
| 并发 | 四路连续批处理，MTP 目标和草稿跨路、跨时间合批；每路 262,144 token，共享 1,048,576 |
| 状态与缓存 | GDN 状态只克隆被接受部分、请求私有状态 arena、CUDA Graph 通过间接描述符读请求状态、多分支写时复制前缀缓存 |
| 内核 | 手写 CUDA（INT8 输出头、FP8 量化、共享 epilogue）与 Triton（GDN、QSA 稀疏注意力、超连接、KV、DFlash） |
| 服务 | 思考预算（low/medium/high）、防重复、工具 Schema 校验、取消、有界队列 |

## 性能

同一台 DGX Spark、同一份转换后的检查点、同一套合成 Agent 负载（多步工具调用，温度 0，关闭思考，每次最多 1024 输出 token）。吞吐为全部输出 token 除以墙钟时间，两次中位数，多路为合计。

| 并发 | Ember | vLLM（开发版） | llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | **40.1** tok/s | 32.8 | 23.1 |
| 2 | **46.4** | 37.2 | 22.5 |
| 4 | **56.7** | 48.2 | 37.5 |

单路代码修复场景的 decode（SSE，扣除首个 chunk 前的时间）：**55.7** / 46.2 / 26.4 tok/s。

注意：三个引擎的草稿器、缓存和上下文上限不同，输出序列也不同；工具为模拟执行；每格重复 2 次。vLLM 是 GB10 开发版，不是最新正式版。这是服务配置对照，不是内核排名。详见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md) 和 [benchmarks/](benchmarks/)。

## 运行要求

- NVIDIA DGX Spark（GB10，SM121），CUDA 13，支持 SM121 的 `nvcc`。
- 容器镜像需含 PyTorch、Triton、`transformers` 和 `nvidia-cutlass-dsl` 4.7.1。引擎还会从 GB10 版 vLLM 加载三个编译算子（`_C` 的 `cutlass_scaled_mm`、`_moe_C` 的 `topk_softmax`、确定性 `persistent_topk`）；不在 `vllm/` 下时用 `ENGINE_NATIVE_KERNEL_DIR` 指定。
- 权重：
  - 目标模型：`nvidia/Qwen3.8-Flash-Next-NVFP4` 转换为本引擎格式（三值 PLE 表；FP8 投影和 INT8 输出头由官方 BF16 重新生成；Top10 + 共享专家配置）；
  - 草稿器：`incoai/Qwen3.8-Flash-Next-DFlash2`，加载时量化为 NVFP4。

  **0.1.0 暂不包含权重转换工具**，目前还不能只凭公开检查点直接运行。

## 快速开始

```bash
# 1. 在 GB10 上编译 CUDA 内核
scripts/build_kernels.sh

# 2. 检查优化能否安装（不加载权重）
SPARK_LAB_DIR=$(mktemp -d) python3 -m serve --check-install   # 在镜像内的 runtime/ 下运行

# 3. 启动服务
IMAGE=... MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... scripts/run_docker.sh
```

容器内监听 8000 端口，请求需带 `Authorization: Bearer <key>`。接口：`/v1/models`、`/v1/chat/completions`（流式与工具）、`/v1/scheduler`。

## 测试

```bash
cd runtime && python3 -m unittest discover -p 'test_*.py'
```

多数测试在 CPU 上运行，但需要镜像中的 PyTorch、Triton 和分词器；需要真实分词器的测试读取 `SPARK_TEST_TOKENIZER_DIR`，未设置时跳过。CPU 测试不能证明 GPU 数值正确或端到端提速。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。第三方代码保留原许可：`engine/vendor/fla/`（Apache-2.0 与 MIT），`moe_upfirst.py`（Apache-2.0，FlashInfer）。本仓库不分发模型权重，权重遵循各自许可。
