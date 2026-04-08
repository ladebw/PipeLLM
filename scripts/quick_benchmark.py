#!/usr/bin/env python3
"""
Quick benchmark for Task 1.6 verification.

This runs a minimal benchmark to verify that the CUDA graph implementation
is working and provides performance improvements.
"""

import sys
import os
import torch
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType
from cuda_graph.output_validation import OutputValidator


def quick_cuda_graph_benchmark():
    """Run a quick benchmark to verify CUDA graph performance."""
    print("Quick CUDA Graph Benchmark")
    print("=" * 50)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create simple computation function
    def simple_computation(x, mask=None, pos=None, past_kv=None):
        # Simple matrix operations to simulate transformer computation
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        output = output @ x
        return output
    
    # Create graph manager
    graph_manager = CUDAGraphManager()
    
    # Create validator
    validator = OutputValidator(absolute_tolerance=1e-5, relative_tolerance=1e-4)
    
    # Test sequence lengths
    test_lengths = [512, 1024]
    results = []
    
    for seq_len in test_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        # Determine graph type
        if seq_len <= 512:
            graph_type = GraphType.SHORT
        else:
            graph_type = GraphType.STANDARD
        
        # Capture graph
        print(f"  Capturing {graph_type.name} graph...")
        capture_success = graph_manager.capture_graph(
            graph_type=graph_type,
            capture_func=simple_computation
        )
        
        if not capture_success:
            print(f"  ✗ Failed to capture graph")
            continue
        
        # Create test input
        batch_size = 1
        hidden_size = 4096
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = [x, mask, pos, None]
        
        # Benchmark eager execution
        print("  Benchmarking eager execution...")
        eager_times = []
        
        # Warmup
        for _ in range(3):
            _ = simple_computation(*inputs)
            torch.cuda.synchronize()
        
        # Measure
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            eager_output = simple_computation(*inputs)
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            eager_times.append((end - start) * 1000)  # Convert to ms
        
        # Benchmark graph execution
        print("  Benchmarking CUDA graph execution...")
        graph_times = []
        
        # Warmup
        for _ in range(3):
            _ = graph_manager.execute_graph(graph_type, inputs)
            torch.cuda.synchronize()
        
        # Measure
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            graph_output = graph_manager.execute_graph(graph_type, inputs)
            torch.cuda.synchronize()
            
            end = time.perf_counter()
            graph_times.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        eager_mean = sum(eager_times) / len(eager_times)
        graph_mean = sum(graph_times) / len(graph_times)
        
        # Calculate speedup
        if graph_mean > 0:
            speedup = eager_mean / graph_mean
            improvement_pct = (speedup - 1) * 100
        else:
            speedup = 0
            improvement_pct = 0
        
        # Validate outputs
        print("  Validating outputs...")
        validation_result = validator.validate_single_forward(
            eager_func=lambda *args: simple_computation(*args),
            graph_func=lambda *args: graph_manager.execute_graph(graph_type, args),
            inputs=inputs,
            sequence_length=seq_len,
            batch_size=batch_size,
            graph_type=graph_type.name
        )
        
        # Store results
        results.append({
            "sequence_length": seq_len,
            "graph_type": graph_type.name,
            "eager_mean_ms": eager_mean,
            "graph_mean_ms": graph_mean,
            "speedup": speedup,
            "improvement_pct": improvement_pct,
            "validation_passed": validation_result.passed,
            "max_error": validation_result.max_absolute_error
        })
        
        # Print results
        print(f"  Results:")
        print(f"    Eager: {eager_mean:.2f} ms")
        print(f"    Graph: {graph_mean:.2f} ms")
        print(f"    Speedup: {speedup:.2f}x ({improvement_pct:.1f}% improvement)")
        print(f"    Validation: {'PASS' if validation_result.passed else 'FAIL'}")
        if not validation_result.passed:
            print(f"    Max error: {validation_result.max_absolute_error:.2e}")
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if not results:
        print("No benchmarks completed successfully.")
        return False
    
    all_passed = all(r["validation_passed"] for r in results)
    avg_improvement = sum(r["improvement_pct"] for r in results) / len(results)
    
    print(f"Tests completed: {len(results)}")
    print(f"All validations passed: {'Yes' if all_passed else 'No'}")
    print(f"Average improvement: {avg_improvement:.1f}%")
    print()
    
    # Check Phase 1 goal
    goal_min = 10.0  # 10% improvement
    goal_max = 15.0  # 15% improvement
    
    if avg_improvement >= goal_min:
        if avg_improvement <= goal_max:
            print(f"✅ Phase 1 goal ACHIEVED: {avg_improvement:.1f}% improvement")
            print(f"   (Target: {goal_min:.1f}% to {goal_max:.1f}%)")
        else:
            print(f"✅ Phase 1 goal EXCEEDED: {avg_improvement:.1f}% improvement")
            print(f"   (Target: {goal_min:.1f}% to {goal_max:.1f}%)")
    else:
        print(f"❌ Phase 1 goal NOT ACHIEVED: {avg_improvement:.1f}% improvement")
        print(f"   (Target: {goal_min:.1f}% to {goal_max:.1f}%)")
    
    print()
    return all_passed and avg_improvement >= goal_min


def main():
    """Main entry point."""
    try:
        success = quick_cuda_graph_benchmark()
        
        if success:
            print("Quick benchmark completed successfully.")
            print("Task 1.6 verification PASSED.")
            return 0
        else:
            print("Quick benchmark completed with issues.")
            print("Task 1.6 verification may need attention.")
            return 1
            
    except Exception as e:
        print(f"Error during quick benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())