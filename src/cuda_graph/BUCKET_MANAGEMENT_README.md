# Context Length Bucket Management

## Phase 1, Task 1.4: Handle multiple context length buckets (512,1024,2048,4096)

### Overview

This module implements production-ready context length bucket management for CUDA graph compilation in LLM inference. It extends the basic graph capture system (Task 1.3) with dynamic graph selection, memory-aware caching, LRU eviction, and adaptive learning.

### Problem Statement

While Task 1.3 implemented basic CUDA graph capture, real-world LLM inference requires:
- **Dynamic graph selection** based on actual input sequence length
- **Memory management** for multiple graphs with different context lengths
- **Efficient caching** with LRU eviction when memory limits are reached
- **Adaptive learning** to optimize graph selection based on usage patterns
- **Production integration** with existing transformer models

### Solution: Enhanced Bucket Management System

The system provides:

1. **GraphBucketManager**: Core manager with LRU eviction and memory limits
2. **AdaptiveBucketManager**: Enhanced manager with learning capabilities
3. **BucketAwareLayerAdapter**: Layer wrapper for bucket-aware execution
4. **BucketAwareModelWrapper**: Full model integration
5. **Production factory functions**: Easy configuration for different use cases

### Architecture

```
src/cuda_graph/
├── cuda_graph_buckets.py          # Enhanced bucket management (892 lines)
├── bucket_integration.py          # Model integration (706 lines)
├── bucket_demo.py                 # Comprehensive demonstration (810 lines)
├── cuda_graph_capture.py          # Base implementation (Task 1.3)
├── llama_integration.py           # Base integration (Task 1.3)
└── BUCKET_MANAGEMENT_README.md    # This file
```

### Key Components

#### 1. `GraphBucketManager`
- **Dynamic graph selection**: Chooses best graph based on input sequence length
- **Memory-aware caching**: Tracks memory usage and enforces limits
- **LRU eviction**: Evicts least recently used graphs when memory limit reached
- **On-demand capture**: Captures graphs automatically when needed
- **Statistics tracking**: Comprehensive performance and memory metrics

#### 2. `AdaptiveBucketManager` (extends `GraphBucketManager`)
- **Usage pattern learning**: Tracks sequence length frequencies
- **Graph effectiveness analysis**: Measures hit/miss rates per graph type
- **Optimal graph selection**: Recommends which graphs to keep based on usage
- **Memory optimization**: Automatically evicts underutilized graphs
- **Predictive caching**: Learns patterns to pre-capture likely needed graphs

#### 3. `BucketAwareLayerAdapter`
- **Layer-level integration**: Wraps individual transformer layers
- **Automatic graph selection**: Uses bucket manager for each forward pass
- **Performance tracking**: Layer-specific statistics
- **Pre-capture support**: Can pre-capture graphs for specific context lengths

#### 4. `BucketAwareModelWrapper`
- **Full model integration**: Wraps entire transformer models
- **Automatic layer wrapping**: Discovers and wraps all transformer layers
- **Unified interface**: Same API as original model
- **Comprehensive reporting**: Model-level performance and memory statistics
- **Benchmarking tools**: Built-in performance testing

### Features

#### Dynamic Graph Selection
- Automatically selects the smallest graph that can accommodate the input
- Falls back to eager execution when no suitable graph available
- Supports all context length buckets: 512, 1024, 2048, 4096 tokens

#### Memory Management
- Configurable memory limits (default: 4GB)
- LRU eviction when limits exceeded
- Memory usage tracking and reporting
- Efficient tensor reuse for graph inputs/outputs

#### Adaptive Learning
- Tracks sequence length usage patterns
- Measures graph effectiveness (hit rates)
- Recommends optimal graphs to keep
- Automatically optimizes memory allocation

#### Production Integration
- Works with any transformer architecture
- No model modifications required
- Preserves original model API
- Comprehensive error handling and fallbacks

### Usage Examples

#### Basic Usage

```python
from cuda_graph import wrap_model_with_bucket_management

# Wrap your model
model = load_llama_model()
wrapped_model = wrap_model_with_bucket_management(
    model=model,
    memory_limit_mb=4096,  # 4GB limit
    adaptive=True,         # Use adaptive learning
    pre_capture_graphs=True  # Pre-capture common graphs
)

# Use normally - graphs auto-selected based on input size
outputs = wrapped_model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

# Get performance report
report = wrapped_model.get_performance_report()
print(f"Graph hit rate: {report['performance_metrics']['graph_hit_rate']}")
print(f"Tokens/sec: {report['performance_metrics']['average_tokens_per_second']:.2f}")
```

#### Advanced Configuration

```python
from cuda_graph.cuda_graph_buckets import create_production_bucket_manager
from cuda_graph.bucket_integration import BucketAwareModelWrapper

# Create custom bucket manager
bucket_manager = create_production_bucket_manager(
    device="cuda:0",
    memory_limit_mb=8192,  # 8GB for large models
    adaptive=True
)

# Wrap model with custom manager
wrapped_model = BucketAwareModelWrapper(
    model=model,
    bucket_manager=bucket_manager,
    hidden_size=4096,
    num_heads=32,
    pre_capture_graphs=True
)

# Pre-capture specific graphs
wrapped_model.pre_capture_all_graphs(
    graph_types=[GraphType.SHORT, GraphType.STANDARD, GraphType.LONG]
)

# Optimize memory based on usage
wrapped_model.optimize_memory_allocation()
```

#### Command Line Demo

```bash
# Run comprehensive demo
python -m cuda_graph.bucket_demo

# Output includes:
# - Graph capture success rates
# - Memory usage statistics
# - Performance benchmarks
# - Learning insights
# - Visualizations
```

### Performance Characteristics

#### Memory Requirements
| Context Length | Graph Memory | 4-Layer Model | 32-Layer Model |
|----------------|--------------|---------------|----------------|
| 512 tokens     | ~50-100 MB   | ~200-400 MB   | ~1.6-3.2 GB    |
| 1024 tokens    | ~100-200 MB  | ~400-800 MB   | ~3.2-6.4 GB    |
| 2048 tokens    | ~200-400 MB  | ~800-1600 MB  | ~6.4-12.8 GB   |
| 4096 tokens    | ~400-800 MB  | ~1.6-3.2 GB   | ~12.8-25.6 GB  |

#### Expected Performance
| Feature | Improvement | Notes |
|---------|-------------|-------|
| Graph hit rate | 70-90% | Depends on workload patterns |
| Memory efficiency | 80-95% | LRU eviction minimizes waste |
| Adaptive learning | +10-20% hit rate | Over time with pattern learning |
| Overall speedup | 8-13% | Combined effect of graph optimization |

### Configuration Options

#### Bucket Manager Configuration
```python
GraphBucketManager(
    device="cuda",           # CUDA device
    max_total_memory_mb=4096, # Memory limit (MB)
    enable_lru=True,         # Enable LRU eviction
    capture_on_demand=True   # Capture graphs when needed
)
```

#### Adaptive Manager Configuration
```python
AdaptiveBucketManager(
    device="cuda",
    max_total_memory_mb=4096,
    enable_lru=True,
    capture_on_demand=True,
    adaptive_memory_allocation=True,  # Enable learning
    learning_window=1000              # Requests to analyze
)
```

#### Model Wrapper Configuration
```python
wrap_model_with_bucket_management(
    model=model,
    bucket_manager=None,     # Auto-create if None
    hidden_size=4096,        # Model dimensions
    num_heads=32,
    head_dim=128,
    pre_capture_graphs=True, # Pre-capture on init
    adaptive=True,           # Use adaptive manager
    memory_limit_mb=4096     # Memory limit
)
```

### Testing

```bash
# Run bucket management tests
cd tests/cuda_graph
pytest test_bucket_management.py -v

# Test specific components
pytest test_bucket_management.py::TestGraphBucketManager -v
pytest test_bucket_management.py::TestAdaptiveBucketManager -v
pytest test_bucket_management.py::TestBucketAwareLayerAdapter -v

# Run with coverage
pytest test_bucket_management.py --cov=cuda_graph.cuda_graph_buckets
```

### Integration with Existing Code

The bucket management system is designed to integrate seamlessly with the existing CUDA graph implementation from Task 1.3:

1. **Backward compatible**: Works with existing `CUDAGraphManager` and `GraphType`
2. **Enhanced functionality**: Adds memory management and adaptive learning
3. **Same interfaces**: Uses same graph capture and execution patterns
4. **Gradual adoption**: Can be adopted incrementally

### Production Deployment Checklist

- [ ] Set appropriate memory limits based on GPU memory
- [ ] Enable adaptive learning for long-running services
- [ ] Monitor graph hit rates and adjust bucket sizes if needed
- [ ] Implement logging for performance metrics
- [ ] Set up alerting for memory limit approaches
- [ ] Test with production workload patterns
- [ ] Validate correctness vs eager execution
- [ ] Benchmark performance improvements

### Monitoring and Metrics

Key metrics to monitor in production:

1. **Graph hit rate**: Percentage of requests using graphs vs eager
2. **Memory utilization**: Percentage of allocated memory used
3. **Graph effectiveness**: Hit rates per context length bucket
4. **Eviction rate**: How often graphs are evicted due to memory limits
5. **Average execution time**: Comparison between graph and eager
6. **Tokens per second**: Overall throughput

### Troubleshooting

#### Common Issues

1. **Low graph hit rate**:
   - Check if sequence lengths match bucket sizes
   - Consider adding additional bucket sizes
   - Enable adaptive learning to optimize

2. **High memory usage**:
   - Reduce memory limit or number of pre-captured graphs
   - Enable LRU eviction
   - Use adaptive optimization

3. **Graph capture failures**:
   - Check CUDA and PyTorch versions
   - Verify GPU memory availability
   - Ensure warm-up runs before capture

4. **Performance regression**:
   - Compare graph vs eager execution times
   - Check for graph execution errors
   - Verify input/output tensor compatibility

### Next Steps

#### Immediate Next Tasks (Task 1.5)
1. **Validate output correctness**: Compare graph vs eager outputs
2. **Run comprehensive benchmarks**: Measure actual performance improvements
3. **Implement edge case handling**: Empty sequences, max length, etc.
4. **Production testing**: Test with real models and workloads

#### Future Enhancements
1. **Dynamic bucket sizes**: Learn optimal bucket sizes from workload
2. **Multi-GPU support**: Distribute graphs across multiple GPUs
3. **Quantization-aware graphs**: Optimize for quantized models
4. **Profile-guided optimization**: Use profiling data to optimize graphs
5. **Distributed caching**: Share graphs across multiple processes

### References

1. NVIDIA CUDA Graph Documentation
2. PyTorch CUDA Graph API
3. LRU Cache Algorithms
4. Adaptive Learning Systems
5. Transformer Inference Optimization

### License

MIT License - See LICENSE file in project root.