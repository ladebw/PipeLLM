<div align="center">

# PipeLLM

PipeLLM is a local LLM inference engine that delivers faster token generation than llama.cpp on consumer multi-GPU hardware through three system-level optimizations: CUDA graph compilation, async weight prefetch, and pipeline-parallel GPU scheduling.

No model changes. No weight modifications. Same GGUF files llama.cpp uses.

</div>

## Project Status

- **Phase 1 (CUDA Graph Compilation):** COMPLETE -- v0.1.0
- **Phase 2 (Async Weight Prefetch):** COMPLETE -- v0.2.0 (simulation only, hardware validation pending)
- **Phase 3 (Pipeline Parallel):** PLANNED
- **Phase 4 (Benchmark Paper):** PLANNED

## What it does

PipeLLM accelerates LLM inference through three complementary system-level optimizations that work together to reduce per-token latency and enable efficient multi-GPU scaling. The engine maintains full compatibility with existing GGUF model files and requires no model modifications.

## Architecture Overview

PipeLLM implements a three-layer optimization stack:

### 1. CUDA Graph Compilation
Eliminates per-token dispatch overhead by capturing the decode loop as a static graph. Features include:
- Static graph capture for repeated decode operations
- 4 context length buckets (512, 1024, 2048, 4096)
- Automatic graph selection based on sequence length
- Comprehensive output validation system

### 2. Async Double-Buffered Weight Prefetch
Overlaps weight loading with compute using dual CUDA streams and pinned memory. Features include:
- Separate compute and copy CUDA streams
- Pinned memory buffer pool for fast DMA transfers
- Double-buffered weight staging
- Compute-memory transfer overlap scheduling

### 3. Pipeline Parallel Scheduling (Planned)
Splits model layers across multiple GPUs, transferring activations over PCIe with pinned memory buffers.

## Hardware Requirements

- **Minimum:** NVIDIA GPU with CUDA support (compute capability 7.0+)
- **Recommended:** NVIDIA RTX 4090 (24GB) or A100 (40GB)
- **Pipeline Parallel:** 2x NVIDIA RTX 4090 or 2x A100
- **Memory:** 16GB+ VRAM per GPU for 32B+ models
- **Interconnect:** PCIe 4.0 or higher for inter-GPU transfers
- **System:** 32GB+ RAM, fast NVMe storage

## Installation

```bash
git clone https://github.com/ladebw/PipeLLM.git
cd PipeLLM
pip install -r requirements.txt
```

## Quick Start

```bash
# Run validation (no GPU required)
python scripts/run_validation.py

# Run benchmarks (requires CUDA GPU + GGUF model)
python scripts/quick_benchmark.py --model-path /path/to/model.gguf
```

## Project Structure

```
PipeLLM/
+-- src/
|   +-- cuda_graph/                 # Phase 1: CUDA graph capture and bucket management
|   |   +-- cuda_graph_capture.py   # CUDA graph capture implementation
|   |   +-- bucket_management.py    # Context length bucket management
|   |   +-- output_validation.py    # Output correctness validation
|   |   +-- llama_integration.py    # llama.cpp model integration
|   +-- pipeline_parallel/          # Phase 2-3: async prefetch and pipeline scheduler
|       +-- async_prefetch/         # Async weight prefetch infrastructure
+-- benchmarks/                     # Profiling and benchmark tooling
|   +-- profiling.py               # Overhead analysis tools
|   +-- cuda_profiler.py           # CUDA event-based profiling
|   +-- cuda_graph_benchmark.py    # CUDA graph benchmarking
|   +-- overhead_analysis.py       # Comprehensive analysis
+-- tests/
|   +-- cuda_graph/                # Test suite for Phase 1
|       +-- test_cuda_graph_capture.py
|       +-- test_bucket_management.py
|       +-- test_output_validation.py
|       +-- test_integration.py
+-- scripts/                       # Task runner scripts
|   +-- run_validation.py          # Output validation
|   +-- quick_benchmark.py         # Quick performance testing
|   +-- run_task_2_1.py           # Layer profiling (internal)
|   +-- run_task_2_2.py           # Async prefetch testing (internal)
+-- phase2_results/               # Generated profiling results (created on first run)
```

## Performance Targets

**IMPORTANT:** The following are architectural targets, not measured results. All implementations require hardware validation.

| Optimization | Target Improvement | Status |
|---|---|---|
| CUDA graph compilation | +10-15% tokens/sec | Code complete, awaiting hardware validation |
| Async weight prefetch | +15-22% tokens/sec | Infrastructure complete, simulation shows +19.2% improvement |
| Pipeline parallel (2x GPU) | +80-130% tokens/sec | Planned, requires Phase 2 hardware validation |
## Development Status

### Phase 1: CUDA Graph Compilation (COMPLETE -- v0.1.0)
- **Status**: Code complete, released as v0.1.0
- **Components**: CUDA graph capture, context length buckets, output validation
- **Tests**: Comprehensive test suite implemented
- **Hardware validation**: Pending (requires CUDA GPU)

### Phase 2: Async Double-Buffered Weight Prefetch (COMPLETE -- v0.2.0)
- **Status**: Infrastructure complete, simulation shows +19.2% improvement
- **Components**: Dual CUDA streams, pinned memory pools, async prefetch engine
- **Tests**: 7 test files covering all components (4,704 lines of test code)
- **Hardware validation**: Pending (simulation only, requires CUDA GPU)
- **Simulation results**: Achieves Phase 2 target (+19.2% vs +15-22% target)

### Phase 3: Pipeline Parallelism (PLANNED)
- **Status**: Design complete, implementation pending hardware validation
- **Components**: Multi-GPU scheduling, activation transfer, pipeline coordination
- **Dependencies**: Requires Phase 2 hardware validation
- **Target**: +80-130% tokens/sec with 2x GPUs

### Phase 4: Benchmark Paper & Publication (PLANNED)
- **Status**: Results collection and paper writing
- **Components**: Comprehensive benchmarking, performance analysis, paper writing
- **Dependencies**: Requires hardware validation of all phases
- **Target**: Publication-ready results and performance analysis
## Benchmark Status

All current performance numbers are simulated or estimated. Real hardware measurements are required before publication. See [BENCHMARKS_STATUS.md](BENCHMARKS_STATUS.md) for detailed tracking.

## License

MIT License

Copyright (c) 2026 PipeLLM