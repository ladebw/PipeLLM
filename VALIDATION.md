# CUDA Graph Output Validation

This directory contains the output validation system for PipeLLM's CUDA graph compilation.

## Overview

The validation system ensures that CUDA graph execution produces identical outputs to eager (baseline) execution, which is critical for maintaining model correctness while achieving performance improvements.

## Quick Start

### Run Quick Validation
```bash
python scripts/run_validation.py
```

### Run Full Validation
```bash
python scripts/validate_cuda_graph.py --context-lengths 512 1024 2048 --output-dir ./validation_results
```

## Key Features

- **Numerical equivalence validation** with configurable tolerance
- **Multiple context length support** (512, 1024, 2048, 4096)
- **Comprehensive error detection** (NaN, infinity, shape mismatches)
- **Detailed reporting** in both human-readable and JSON formats
- **Integration** with existing CUDA graph infrastructure

## Usage in Code

```python
from cuda_graph.output_validation import OutputValidator

# Create validator
validator = OutputValidator()

# Validate single forward pass
result = validator.validate_single_forward(
    eager_func=eager_func,
    graph_func=graph_func,
    inputs=inputs,
    sequence_length=1024
)

# Validate multiple context lengths
results = validator.validate_multiple_context_lengths(
    eager_func=eager_func,
    graph_func=graph_func,
    context_lengths=[512, 1024, 2048, 4096]
)
```

## Validation Results

Results include:
- Pass/fail status
- Maximum and mean absolute/relative errors
- Number of tensors and elements compared
- Detailed error information for failures
- Warnings for potential issues

## Integration

The validation system is integrated with:
- CUDA graph capture and execution
- Context length bucket management
- Model wrapper infrastructure
- Existing test suite

## Requirements

- PyTorch with CUDA support
- Python 3.8+
- Existing PipeLLM CUDA graph infrastructure

## Notes

For detailed implementation documentation and internal reports, refer to the private documentation in the LLM directory.