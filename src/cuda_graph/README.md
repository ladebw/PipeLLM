# CUDA Graph Compilation for LLM Decode Loops

## Phase 1, Task 1.3: Implement static CUDA graph capture for decode loop

### Overview

This module implements CUDA graph compilation to eliminate per-token dispatch overhead in LLM inference. By capturing the computational graph of decode operations statically, we can achieve 10-15% faster token generation compared to eager execution.

### Problem Statement

Traditional LLM inference suffers from per-token dispatch overhead:
- Each token generation requires launching multiple CUDA kernels
- Kernel launch overhead accumulates across layers and tokens
- Synchronization between operations adds latency
- Memory transfers between host and device create bottlenecks

### Solution: Static CUDA Graph Capture

CUDA graphs allow us to:
1. **Capture** the entire computational graph of a decode operation
2. **Compile** it into a single, optimized executable graph
3. **Execute** the graph repeatedly with minimal overhead
4. **Reuse** the graph for tokens with similar computational patterns

### Architecture

```
src/cuda_graph/
├── cuda_graph_capture.py    # Core graph capture and management
├── llama_integration.py     # Integration with llama.cpp-style models
├── __main__.py             # Command-line interface
├── demo.py                 # Comprehensive demonstration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Key Components

#### 1. `CUDAGraphManager`
- Manages graph capture, storage, and execution
- Handles multiple context length buckets (512, 1024, 2048, 4096)
- Provides fallback to eager execution when graphs don't match
- Tracks performance statistics and hit rates

#### 2. `LlamaLayerGraphAdapter`
- Wraps individual transformer layers for graph capture
- Integrates with llama.cpp-style model architectures
- Manages input/output tensor preparation for graphs
- Handles past key/value cache for autoregressive decoding

#### 3. `LlamaModelGraphWrapper`
- Wraps entire transformer models with graph support
- Manages graph capture across all layers
- Provides unified interface for graph-enabled inference
- Includes benchmarking and analysis tools

### Usage

#### Basic Graph Capture

```python
from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType

# Initialize graph manager
graph_manager = CUDAGraphManager()

# Define computation to capture
def layer_forward(x, mask, pos, past_kv):
    # Your transformer layer computation
    return output, new_past_kv, attention_weights

# Capture graph for standard context length
graph_instance = graph_manager.capture_graph(
    graph_type=GraphType.STANDARD,  # 1024 tokens
    capture_func=layer_forward
)

# Execute captured graph
outputs = graph_manager.execute_graph(
    graph_type=GraphType.STANDARD,
    inputs=[x, mask, pos, past_kv]
)
```

#### Model Integration

```python
from cuda_graph.llama_integration import wrap_model_with_cuda_graphs

# Wrap your model with CUDA graph support
wrapped_model = wrap_model_with_cuda_graphs(
    model=your_llama_model,
    capture_seq_lens=[512, 1024, 2048, 4096]
)

# Capture graphs for all layers
wrapped_model.capture_all_graphs()

# Use the model normally - graphs will be used automatically
outputs = wrapped_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values
)
```

#### Command Line Interface

```bash
# Run demo
python -m cuda_graph demo

# Benchmark context length buckets
python -m cuda_graph benchmark

# With verbose output
python -m cuda_graph demo --verbose
```

### Performance Targets

| Context Length | Target Improvement | Expected Speedup |
|----------------|-------------------|------------------|
| 512 tokens     | 10-12%            | 1.10-1.12x       |
| 1024 tokens    | 12-14%            | 1.12-1.14x       |
| 2048 tokens    | 13-15%            | 1.13-1.15x       |
| 4096 tokens    | 14-16%            | 1.14-1.16x       |

### Memory Considerations

- Each captured graph requires GPU memory
- Memory usage scales with context length
- Typical usage per graph:
  - 512 tokens: ~50-100 MB
  - 1024 tokens: ~100-200 MB  
  - 2048 tokens: ~200-400 MB
  - 4096 tokens: ~400-800 MB

### Testing

```bash
# Run unit tests
cd tests/cuda_graph
pytest test_cuda_graph_capture.py -v

# Run with CUDA (requires GPU)
pytest test_cuda_graph_capture.py::TestGraphCapture -v
```

### Dependencies

- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 11.8+
- NVIDIA GPU with compute capability 7.0+
- Python 3.8+

### Implementation Status

- [x] Core graph capture infrastructure
- [x] Context length bucket management
- [x] Integration with llama.cpp-style models
- [x] Benchmarking and analysis tools
- [x] Comprehensive demo and visualization
- [x] Unit tests for core functionality
- [ ] Integration with actual GGUF model loader
- [ ] Production-ready error handling
- [ ] Multi-GPU support

### Next Steps (Task 1.4)

1. **Handle multiple context length buckets in production**
   - Dynamic graph selection based on input size
   - Memory-aware graph management
   - LRU cache for graph instances

2. **Validate output correctness**
   - Compare graph vs eager outputs
   - Numerical stability testing
   - Edge case handling

3. **Performance optimization**
   - Graph fusion opportunities
   - Memory layout optimization
   - Stream concurrency

### References

1. NVIDIA CUDA Graph Documentation
2. PyTorch CUDA Graph API
3. llama.cpp Architecture
4. Transformer Inference Optimization Papers

### License

MIT License - See LICENSE file in project root.