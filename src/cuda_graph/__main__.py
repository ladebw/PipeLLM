"""
Main entry point for CUDA Graph Compilation module.

This module provides command-line interface for testing and benchmarking
CUDA graph compilation for LLM decode loops.
"""

import argparse
import sys
import torch
import logging
from typing import Dict, Any

from .cuda_graph_capture import CUDAGraphManager, GraphType, benchmark_graph_vs_eager
from .llama_integration import wrap_model_with_cuda_graphs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_cuda_graph_capture():
    """Demo basic CUDA graph capture functionality."""
    print("=" * 60)
    print("CUDA Graph Capture Demo")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Initialize graph manager
    graph_manager = CUDAGraphManager()
    
    # Create a simple computation to capture
    def simple_computation(x, y):
        # Simple matrix operations
        z = x @ y.T
        z = torch.relu(z)
        z = z.mean(dim=1, keepdim=True)
        return z
    
    # Capture graph for standard context length
    try:
        print("\n1. Capturing CUDA graph for standard context length (1024)...")
        graph_instance = graph_manager.capture_graph(
            graph_type=GraphType.STANDARD,
            capture_func=simple_computation,
            # Inputs will be created by capture_graph
        )
        
        print(f"   ✓ Graph captured in {graph_instance.capture_time:.3f}s")
        print(f"   ✓ Estimated memory: {graph_instance.memory_usage / 1024**2:.2f} MB")
        
        # Test graph execution
        print("\n2. Testing graph execution...")
        
        # Create test inputs
        batch_size = 1
        seq_len = 1024
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        y = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        
        # Execute graph
        outputs = graph_manager.execute_graph(GraphType.STANDARD, [x, y])
        
        print(f"   ✓ Graph executed successfully")
        print(f"   ✓ Output shape: {outputs[0].shape}")
        
        # Benchmark
        print("\n3. Benchmarking graph vs eager execution...")
        
        def eager_func(x, y):
            return simple_computation(x, y)
        
        results = benchmark_graph_vs_eager(
            graph_manager=graph_manager,
            eager_func=eager_func,
            seq_len=seq_len,
            num_iterations=50
        )
        
        print(f"   ✓ Eager execution: {results['eager_avg_ms']:.2f} ± {results['eager_std_ms']:.2f} ms")
        
        if 'graph_avg_ms' in results:
            print(f"   ✓ Graph execution: {results['graph_avg_ms']:.2f} ± {results['graph_std_ms']:.2f} ms")
            print(f"   ✓ Speedup: {results['speedup']:.2f}x")
        
        # Show statistics
        print("\n4. Statistics:")
        stats = graph_manager.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


def benchmark_context_lengths():
    """Benchmark different context length buckets."""
    print("=" * 60)
    print("Context Length Bucket Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return
    
    graph_manager = CUDAGraphManager()
    
    # Define a simple computation
    def layer_forward(x, mask, pos, past_kv):
        # Simulate a transformer layer forward pass
        # Attention
        attn = x @ x.transpose(-2, -1) / (x.shape[-1] ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        
        # Output projection
        output = attn @ x
        
        # FFN (simplified)
        output = torch.relu(output)
        output = output @ torch.randn_like(output)
        
        # Residual connection
        output = output + x
        
        return output, past_kv, None
    
    # Test different context lengths
    context_lengths = [512, 1024, 2048, 4096]
    results = []
    
    for seq_len in context_lengths:
        print(f"\nBenchmarking seq_len={seq_len}...")
        
        # Determine graph type
        graph_type = None
        for gt in GraphType:
            if gt.value >= seq_len:
                graph_type = gt
                break
        
        if graph_type is None:
            print(f"  Skipping - no suitable graph type")
            continue
        
        # Capture graph if not already captured
        if graph_type not in graph_manager.graphs:
            print(f"  Capturing graph for {graph_type.name}...")
            try:
                graph_manager.capture_graph(
                    graph_type=graph_type,
                    capture_func=layer_forward,
                )
            except Exception as e:
                print(f"  Failed to capture graph: {e}")
                continue
        
        # Benchmark
        try:
            benchmark_results = benchmark_graph_vs_eager(
                graph_manager=graph_manager,
                eager_func=layer_forward,
                seq_len=seq_len,
                num_iterations=30
            )
            
            results.append(benchmark_results)
            
            print(f"  Eager: {benchmark_results['eager_avg_ms']:.2f} ms")
            if 'graph_avg_ms' in benchmark_results:
                print(f"  Graph: {benchmark_results['graph_avg_ms']:.2f} ms")
                print(f"  Speedup: {benchmark_results['speedup']:.2f}x")
            
        except Exception as e:
            print(f"  Benchmark failed: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    for result in results:
        print(f"\nseq_len={result['seq_len']}:")
        print(f"  Eager: {result['eager_avg_ms']:.2f} ± {result['eager_std_ms']:.2f} ms")
        
        if 'graph_avg_ms' in result:
            print(f"  Graph: {result['graph_avg_ms']:.2f} ± {result['graph_std_ms']:.2f} ms")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Graph Type: {result['graph_type']}")
    
    print("\n" + "=" * 60)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="CUDA Graph Compilation for LLM Decode Loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demo           - Run CUDA graph capture demo
  %(prog)s benchmark      - Benchmark context length buckets
  %(prog)s --help         - Show this help message
        """
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'benchmark'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        help='CUDA device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check CUDA availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        print("Available devices:", torch.cuda.device_count())
        return 1
    
    # Execute command
    if args.command == 'demo':
        demo_cuda_graph_capture()
    elif args.command == 'benchmark':
        benchmark_context_lengths()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())