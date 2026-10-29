# Ember

面向 **NVIDIA DGX Spark（GB10，SM121）** 的 **Qwen3.8 Flash Next** 推理引擎，提供 OpenAI 兼容的 HTTP/SSE 接口，支持工具调用和思考预算。

[English](README.md)

> **状态：0.1.2，实验性。** 只支持一个模型、一种 GPU。测试结果见 [验证记录](docs/VALIDATION.md)；完整质量评测和四路满 262K 上下文尚未测试。

## 主要技术

| 方向 | 内容 |
| --- | --- |
| 投机解码 | 原生 MTP 头 + DFlash2 块草稿器（NVFP4 权重、融合 Triton 内核），成本模型在两者间选择，单路 8 行验证 |
| MoE | SM121 W4A4 路由专家 MoE（NVFP4 激活 × NVFP4 权重，CuTe DSL），确定性 tactic 表，router/indexer 逐行 GEMV 合并 |
| 并发 | 四路连续批处理，MTP 目标和草稿跨路、跨时间合批；上下文上限每路 262,144 token、共享 1,048,576 |
| 状态与缓存 | GDN 状态只克隆被接受部分、请求私有状态 arena、CUDA Graph 通过间接描述符读请求状态、多分支写时复制前缀缓存 |
| 内核 | CUDA（INT8 输出头、FP8 量化、共享 epilogue）与 Triton（GDN、QSA 稀疏注意力、超连接、KV、DFlash） |
| 服务 | 思考预算（low/medium/high）、防重复、工具 Schema 校验、取消、有界队列 |

## 性能

同一台 DGX Spark、同一份转换后的检查点、同一套合成 Agent 负载（多步工具调用，温度 0，关闭思考，每次最多 1024 输出 token）。吞吐为全部输出 token 除以墙钟时间，两次中位数，多路为合计。

| 并发 | Ember | vLLM（开发版） | llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | **40.1** tok/s | 32.8 | 23.1 |
| 2 | **46.4** | 37.2 | 22.5 |
| 4 | **56.7** | 48.2 | 37.5 |

单路代码修复场景的 decode（SSE，扣除首个 chunk 前的时间）：**55.7** / 46.2 / 26.4 tok/s。

三个引擎的草稿器、缓存和上下文上限不同，输出序列也不同；工具为模拟执行；每格跑 2 次；vLLM 为 GB10 开发版。详见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md) 和 [benchmarks/](benchmarks/)。

## 运行要求

- NVIDIA DGX Spark（GB10，SM121），CUDA 13，支持 SM121 的 `nvcc`。
- 容器镜像需含 PyTorch、Triton、`transformers` 和 `nvidia-cutlass-dsl` 4.7.1。引擎还会从 GB10 版 vLLM 加载三个编译算子（`_C` 的 `cutlass_scaled_mm`、`_moe_C` 的 `topk_softmax`、确定性 `persistent_topk`）；不在 `vllm/` 下时用 `ENGINE_NATIVE_KERNEL_DIR` 指定。
- 权重：
  - 目标模型：`nvidia/Qwen3.8-Flash-Next-NVFP4` 转换为本引擎格式（三值 PLE 表；FP8 投影和 INT8 输出头由官方 BF16 重新生成；Top10 + 共享专家配置）；
  - 草稿器：`incoai/Qwen3.8-Flash-Next-DFlash2`，加载时量化为 NVFP4。

  已提供 [`tools/convert_checkpoint.py`](tools/convert_checkpoint.py)，可从官方检查点直接生成引擎格式，步骤与量化细节见 [权重转换](docs/WEIGHTS.md)。

不提供预构建镜像。参考版本、可选 Dockerfile 和环境检查脚本见 [环境说明](docs/ENVIRONMENT.md)。

## 快速开始

```bash
export IMAGE='your-compatible-GB10-image:tag'
# 在镜像中编译 CUDA 内核与独立 QSA 算子
docker run --rm -v "$PWD:/repo" -w /repo --entrypoint bash "$IMAGE" scripts/build_kernels.sh
# 检查依赖与库文件，不加载权重
docker run --rm -v "$PWD:/repo:ro" -w /repo --entrypoint python3 "$IMAGE" scripts/check_environment.py
# 先按 docs/WEIGHTS.md 转换目标检查点，再启动
MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... bash scripts/run_docker.sh
```

启动器会检查 GPU 空闲、容器名未被占用，然后为该容器启动内存守卫：启动时需要
至少 110 GiB 可用内存，可用内存低于 20 GiB 或 swap 增长过快时停止该容器。权重只读挂载。

容器内监听 8000 端口，请求需带 `Authorization: Bearer <key>`。接口：`/v1/models`、`/v1/chat/completions`（流式与工具）、`/v1/scheduler`。

### 可选整树投机

默认继续使用 MTP。在启动时选择实验性的 best-first 整树后端：

```bash
SPECULATION=tree MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... bash scripts/run_docker.sh
```

下次启动设为 `SPECULATION=mtp` 可恢复原后端。直接运行入口时，可设置
`SPARK_SPECULATION=tree` 或传入 `--speculation tree`，同时保留
`SPARK_EXECUTION_MODE=continuous_mtp` 和原模型环境。`/health` 与 `/v1/scheduler`
会显示所选模式。

树模式保留流式、工具、思考控制、成对 prefill 缓存复用和取消流程。多请求使用私有
状态合作式逐路执行，尚无 GPU 验证合批；四路满上下文容量未验收。
详见[实现与验证记录](docs/TREE-VERIFICATION.md)。

## 测试

```bash
bash scripts/test.sh
```

先在兼容环境安装 `requirements-test.txt`；pytest 同时收集 unittest 类和函数测试。多数测试在 CPU 上运行，但需要镜像中的 PyTorch、Triton 和分词器；需要真实分词器的测试读取 `SPARK_TEST_TOKENIZER_DIR`，未设置时跳过。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。第三方代码保留原许可：`engine/vendor/fla/`（Apache-2.0 与 MIT），`moe_upfirst.py`（Apache-2.0，FlashInfer）。本仓库不分发模型权重，权重遵循各自许可。
