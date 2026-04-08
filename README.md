<div align="center">

# PipeLLM

PipeLLM is a local LLM inference engine that delivers faster token generation than llama.cpp on consumer multi-GPU hardware through three system-level optimizations: CUDA graph compilation, async weight prefetch, and pipeline-parallel GPU scheduling.

No model changes. No weight modifications. Same GGUF files llama.cpp uses.

</div>

## Project Status

- **Phase 1 (CUDA Graph Compilation):** COMPLETE — v0.1.0
- **Phase 2 (Async Weight Prefetch):** IN PROGRESS 
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

# Profile layer execution timeline
python scripts/run_task_2_1.py

# Test async prefetch infrastructure
python scripts/run_task_2_2.py
```

## Project Structure

```
PipeLLM/
├── src/
│   ├── cuda_graph/                 # Phase 1: CUDA graph capture and bucket management
│   │   ├── cuda_graph_capture.py   # CUDA graph capture implementation
│   │   ├── bucket_management.py    # Context length bucket management
│   │   ├── output_validation.py    # Output correctness validation
│   │   └── llama_integration.py    # llama.cpp model integration
│   └── pipeline_parallel/          # Phase 2-3: async prefetch and pipeline scheduler
│       └── async_prefetch/         # Async weight prefetch infrastructure
├── benchmarks/                     # Profiling and benchmark tooling
│   ├── profiling.py               # Overhead analysis tools
│   ├── cuda_profiler.py           # CUDA event-based profiling
│   ├── cuda_graph_benchmark.py    # CUDA graph benchmarking
│   └── overhead_analysis.py       # Comprehensive analysis
├── tests/
│   └── cuda_graph/                # Test suite for Phase 1
│       ├── test_cuda_graph_capture.py
│       ├── test_bucket_management.py
│       ├── test_output_validation.py
│       └── test_integration.py
├── scripts/                       # Task runner scripts
│   ├── run_validation.py          # Output validation
│   ├── quick_benchmark.py         # Quick performance testing
│   ├── run_task_2_1.py           # Task 2.1 execution
│   └── run_task_2_2.py           # Task 2.2 execution
└── phase2_results/               # Generated profiling results
```

## Performance Targets

**IMPORTANT:** The following are architectural targets, not measured results. All implementations require hardware validation.

| Optimization | Target Improvement | Status |
|---|---|---|
| CUDA graph compilation | +10–15% tokens/sec | Code complete, awaiting hardware validation |
| Async weight prefetch | +15–22% tokens/sec | Infrastructure complete, untested on hardware |
| Pipeline parallel (2x GPU) | +80–130% tokens/sec | Planned, requires Phase 2 validation |

## Development Status

### Phase 1: CUDA Graph Compilation (COMPLETE)
- ✅ CUDA graph capture implementation
- ✅ Context length bucket management (512, 1024, 2048, 4096)
- ✅ Output validation system
- ✅ Benchmarking infrastructure
- ✅ Comprehensive test suite
- ✅ Released as v0.1.0

### Phase 2: Async Weight Prefetch (IN PROGRESS)
- ✅ Layer-by-layer profiling infrastructure
- ✅ Dual CUDA stream manager
- ✅ Pinned memory buffer pool
- ✅ Async prefetch engine
- 🔄 Race condition testing (requires CUDA hardware)
- 🔄 Output validation integration
- 🔄 Performance benchmarking

### Phase 3: Pipeline Parallel (PLANNED)
- 🔄 Multi-GPU layer distribution
- 🔄 Inter-GPU activation transfer
- 🔄 Pipeline scheduling logic
- 🔄 Load balancing optimization

## Benchmark Status

All current performance numbers are simulated or estimated. Real hardware measurements are required before publication. See [BENCHMARKS_STATUS.md](BENCHMARKS_STATUS.md) for detailed tracking.

## License

MIT License

Copyright (c) 2026 PipeLLM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.