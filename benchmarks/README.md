# Benchmarking Harness

This directory contains the benchmarking harness for PipeLLM vs llama.cpp comparison.

## Overview

The benchmarking harness is designed to:
1. Automatically set up llama.cpp as a baseline
2. Run standardized benchmarks across different configurations
3. Collect performance metrics (tokens/sec, execution time)
4. Generate comparative analysis and visualizations

## Structure

- `setup_llamacpp.py` - Downloads and compiles llama.cpp
- `benchmark_config.py` - Configuration for models and benchmarks
- `run_benchmark.py` - Main benchmarking script
- `analyze_results.py` - Analysis and visualization tools
- `requirements.txt` - Python dependencies for analysis

## Quick Start

### 1. Setup llama.cpp baseline

```bash
python setup_llamacpp.py
```

### 2. Install analysis dependencies

```bash
pip install -r requirements.txt
```

### 3. Run benchmarks

```bash
python run_benchmark.py --hardware single_4090
```

### 4. Analyze results

```bash
python analyze_results.py --results-dir benchmark_results --output-dir analysis
```

## Benchmark Configurations

The system tests multiple configurations:

### Model Configurations
- 32B Q4_K_M with 256/64 tokens (short)
- 32B Q4_K_M with 512/128 tokens (standard)
- 70B Q4_K_M with 256/64 tokens (short)

### Hardware Configurations
- Single RTX 4090 (24GB VRAM)
- Dual RTX 4090 (48GB VRAM total)

## Output

Benchmarks generate:
- JSON files with raw results in `benchmark_results/`
- CSV summary files
- PNG plots comparing performance
- Markdown reports with analysis

## Notes

- PipeLLM results are currently placeholders until the engine is implemented
- The harness is designed to be extended as PipeLLM development progresses
- All configurations are defined in `benchmark_config.py` for easy modification