# PipeLLM

PipeLLM is a local LLM inference engine that delivers faster token generation than llama.cpp on consumer multi-GPU hardware through three system-level optimizations: CUDA graph compilation, async weight prefetch, and pipeline-parallel GPU scheduling.

No model changes. No weight modifications. Same GGUF files llama.cpp uses.

## What problem it solves

- Eliminates per-token dispatch overhead with CUDA graph compilation
- Overlaps compute and memory transfer with async weight prefetch
- Enables efficient multi-GPU scaling with pipeline parallel scheduling

## Hardware requirements

- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- Multi-GPU setup for pipeline parallel scaling
- 16GB+ VRAM per GPU for 32B+ models
- PCIe 4.0 or higher for inter-GPU transfers

## Coming soon

- Phase 1: CUDA Graph Compilation
- Phase 2: Async Double-Buffered Weight Prefetch
- Phase 3: Pipeline Parallel Scheduler
- Phase 4: Benchmark Paper and Publication

## License

MIT License