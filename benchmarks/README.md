# Benchmarking and Profiling Tools

This directory contains tools for benchmarking, profiling, and analyzing LLM inference performance.

## Overview

The tools are organized into three main categories:

1. **Benchmarking Harness**: Compare PipeLLM vs llama.cpp performance
2. **Overhead Analysis**: Measure per-token overhead breakdown (Phase 1, Task 1.2)
3. **CUDA Profiling**: Low-level CUDA overhead measurements

## Quick Start

### 1. Setup llama.cpp baseline

```bash
python setup_llamacpp.py
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# For CUDA profiling, also install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run per-token overhead analysis (Task 1.2)

```bash
# Direct profiling (requires GGUF model)
python profiling.py --model-path /path/to/model.gguf --runs 10

# CUDA microbenchmarks
python cuda_profiler.py

# Comprehensive analysis
python overhead_analysis.py --results-dir profiling_results --output-dir overhead_analysis
```

### 4. Run full benchmarks

```bash
python run_benchmark.py --hardware single_4090
python analyze_results.py --results-dir benchmark_results --output-dir analysis
```

## Tools Overview

### Benchmarking Harness
- `setup_llamacpp.py` - Downloads and compiles llama.cpp
- `benchmark_config.py` - Configuration for models and benchmarks
- `run_benchmark.py` - Main benchmarking script
- `analyze_results.py` - Analysis and visualization tools

### Overhead Analysis (Task 1.2)
- `profiling.py` - Direct llama.cpp profiling with timing breakdown
- `cuda_profiler.py` - CUDA microbenchmarks for low-level overhead
- `overhead_analysis.py` - Comprehensive analysis combining both methods
- `OVERHEAD_ANALYSIS.md` - Detailed documentation

### Output Directories
- `benchmark_results/` - Benchmark outputs
- `profiling_results/` - Profiling outputs
- `cuda_benchmarks/` - CUDA benchmark outputs
- `analysis/` - Analysis reports and visualizations
- `overhead_analysis/` - Overhead analysis reports

## Benchmark Configurations

### Model Configurations
- 32B Q4_K_M with 256/64 tokens (short)
- 32B Q4_K_M with 512/128 tokens (standard)
- 70B Q4_K_M with 256/64 tokens (short)

### Hardware Configurations
- Single RTX 4090 (24GB VRAM)
- Dual RTX 4090 (48GB VRAM total)

## Expected Overhead Breakdown

Based on preliminary analysis (Task 1.2):

| Component | Percentage | Optimization Target |
|-----------|------------|---------------------|
| Kernel Execution | 40-60% | Actual computation |
| Memory Transfers | 20-30% | Async weight prefetch |
| Dispatch Overhead | 10-15% | CUDA graph compilation |
| Synchronization | 5-10% | Pipeline parallelism |
| Other | 5% | Various optimizations |

## Notes

- **Phase 1 Focus**: Currently implementing overhead analysis (Task 1.2)
- **Placeholder Results**: PipeLLM results are placeholders until engine implementation
- **Hardware Dependence**: Results vary based on GPU model and configuration
- **Validation Required**: Analysis provides guidance; actual implementation requires validation

## Documentation

- `README.md` - This file (overview)
- `OVERHEAD_ANALYSIS.md` - Detailed overhead analysis documentation
- Generated reports in output directories