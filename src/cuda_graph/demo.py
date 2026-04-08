"""
Demo script for CUDA Graph Compilation (Phase 1, Task 1.3).

This script demonstrates the implementation of static CUDA graph capture
for LLM decode loops to eliminate per-token dispatch overhead.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .cuda_graph_capture import (
    CUDAGraphManager,
    GraphType,
    benchmark_graph_vs_eager,
    create_context_length_buckets
)

# Set up plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]


@dataclass
class DemoResults:
    """Container for demo results."""
    graph_capture_times: Dict[GraphType, float]
    benchmark_results: List[Dict[str, Any]]
    memory_usage: Dict[GraphType, int]
    statistics: Dict[str, Any]


def run_cuda_graph_demo() -> DemoResults:
    """
    Run comprehensive CUDA graph compilation demo.
    
    This demo shows:
    1. Graph capture for different context lengths
    2. Performance comparison vs eager execution
    3. Memory usage analysis
    4. Statistical analysis of improvements
    """
    print("=" * 70)
    print("CUDA GRAPH COMPILATION DEMO - Phase 1, Task 1.3")
    print("=" * 70)
    print("Goal: Eliminate per-token dispatch overhead through static CUDA graph capture")
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system")
        return None
    
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"Hardware: {device_name}")
    print(f"CUDA Version: {cuda_version}")
    print()
    
    # Initialize results
    results = DemoResults(
        graph_capture_times={},
        benchmark_results=[],
        memory_usage={},
        statistics={}
    )
    
    # Initialize graph manager
    print("Initializing CUDA Graph Manager...")
    graph_manager = CUDAGraphManager()
    print("✓ Graph manager initialized")
    print()
    
    # Define a realistic transformer layer computation
    def transformer_layer_forward(x, attention_mask, position_ids, past_key_value):
        """
        Simulate a transformer layer forward pass.
        
        This includes:
        - Layer normalization
        - Attention computation
        - Feed-forward network
        - Residual connections
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Simulate layer norm
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + 1e-5)
        
        # Simulate attention (simplified)
        # QKV projections
        q = x_norm @ torch.randn(hidden_size, hidden_size, device=x.device)
        k = x_norm @ torch.randn(hidden_size, hidden_size, device=x.device)
        v = x_norm @ torch.randn(hidden_size, hidden_size, device=x.device)
        
        # Attention scores
        scores = q @ k.transpose(-2, -1) / (hidden_size ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Attention output
        attn_output = attn_weights @ v
        
        # Output projection
        attn_output = attn_output @ torch.randn(hidden_size, hidden_size, device=x.device)
        
        # Residual connection
        x = x + attn_output
        
        # Feed-forward network (simplified)
        ff1 = x @ torch.randn(hidden_size, hidden_size * 4, device=x.device)
        ff1 = torch.relu(ff1)
        ff2 = ff1 @ torch.randn(hidden_size * 4, hidden_size, device=x.device)
        
        # Final residual
        x = x + ff2
        
        # Simulate past key/value update
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # In reality, we'd concatenate new keys/values
            new_past_key = torch.cat([past_key, k.unsqueeze(2)], dim=2)
            new_past_value = torch.cat([past_value, v.unsqueeze(2)], dim=2)
            past_key_value = (new_past_key, new_past_value)
        
        return x, past_key_value, attn_weights
    
    # Part 1: Graph Capture for Different Context Lengths
    print("PART 1: Graph Capture for Context Length Buckets")
    print("-" * 50)
    
    context_buckets = create_context_length_buckets()
    
    for graph_type, max_tokens in context_buckets.items():
        print(f"\nCapturing graph for {graph_type.name} context ({max_tokens} tokens)...")
        
        try:
            start_time = time.time()
            
            graph_instance = graph_manager.capture_graph(
                graph_type=graph_type,
                capture_func=transformer_layer_forward,
            )
            
            capture_time = time.time() - start_time
            
            results.graph_capture_times[graph_type] = capture_time
            results.memory_usage[graph_type] = graph_instance.memory_usage
            
            print(f"  ✓ Capture time: {capture_time:.3f}s")
            print(f"  ✓ Memory usage: {graph_instance.memory_usage / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.graph_capture_times[graph_type] = -1
            results.memory_usage[graph_type] = -1
    
    print("\n" + "=" * 70)
    
    # Part 2: Performance Benchmarking
    print("\nPART 2: Performance Benchmarking")
    print("-" * 50)
    
    # Test different sequence lengths
    test_seq_lens = [256, 512, 1024, 1536, 2048, 3072, 4096]
    
    for seq_len in test_seq_lens:
        print(f"\nBenchmarking seq_len={seq_len}...")
        
        try:
            benchmark_result = benchmark_graph_vs_eager(
                graph_manager=graph_manager,
                eager_func=transformer_layer_forward,
                seq_len=seq_len,
                num_iterations=30
            )
            
            results.benchmark_results.append(benchmark_result)
            
            print(f"  Eager: {benchmark_result['eager_avg_ms']:.2f} ms")
            
            if 'graph_avg_ms' in benchmark_result:
                speedup = benchmark_result['speedup']
                print(f"  Graph: {benchmark_result['graph_avg_ms']:.2f} ms")
                print(f"  Speedup: {speedup:.2f}x")
                
                if speedup > 1.0:
                    improvement = (speedup - 1.0) * 100
                    print(f"  Improvement: +{improvement:.1f}%")
            else:
                print(f"  Graph: No suitable graph available")
                
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
    
    print("\n" + "=" * 70)
    
    # Part 3: Statistical Analysis
    print("\nPART 3: Statistical Analysis")
    print("-" * 50)
    
    # Calculate overall statistics
    stats = graph_manager.get_stats()
    results.statistics = stats
    
    print(f"\nExecution Statistics:")
    print(f"  Graph executions: {stats['graph_executions']}")
    print(f"  Eager executions: {stats['eager_executions']}")
    
    total_executions = stats['graph_executions'] + stats['eager_executions']
    if total_executions > 0:
        graph_hit_rate = stats['graph_hit_rate'] * 100
        print(f"  Graph hit rate: {graph_hit_rate:.1f}%")
    
    # Calculate average speedup from benchmarks
    speedups = []
    for result in results.benchmark_results:
        if 'speedup' in result:
            speedups.append(result['speedup'])
    
    if speedups:
        avg_speedup = np.mean(speedups)
        std_speedup = np.std(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)
        
        print(f"\nSpeedup Analysis ({len(speedups)} benchmarks):")
        print(f"  Average: {avg_speedup:.2f}x")
        print(f"  Std Dev: {std_speedup:.2f}x")
        print(f"  Range: {min_speedup:.2f}x - {max_speedup:.2f}x")
        
        avg_improvement = (avg_speedup - 1.0) * 100
        print(f"  Average Improvement: +{avg_improvement:.1f}%")
    
    # Memory analysis
    print(f"\nMemory Usage Analysis:")
    for graph_type, memory in results.memory_usage.items():
        if memory > 0:
            print(f"  {graph_type.name}: {memory / 1024**2:.1f} MB")
    
    print("\n" + "=" * 70)
    
    # Part 4: Visualization
    print("\nPART 4: Generating Visualizations")
    print("-" * 50)
    
    try:
        generate_visualizations(results)
        print("✓ Visualizations generated successfully")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    return results


def generate_visualizations(results: DemoResults):
    """Generate visualizations from demo results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Graph Capture Times
    ax1 = axes[0, 0]
    if results.graph_capture_times:
        graph_types = list(results.graph_capture_times.keys())
        capture_times = [results.graph_capture_times[gt] for gt in graph_types]
        context_lengths = [gt.value for gt in graph_types]
        
        bars = ax1.bar(range(len(graph_types)), capture_times)
        ax1.set_xlabel('Context Length')
        ax1.set_ylabel('Capture Time (s)')
        ax1.set_title('CUDA Graph Capture Time by Context Length')
        ax1.set_xticks(range(len(graph_types)))
        ax1.set_xticklabels([f'{gt.name}\n({gt.value})' for gt in graph_types])
        
        # Add value labels on bars
        for bar, time_val in zip(bars, capture_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 2. Performance Speedup
    ax2 = axes[0, 1]
    if results.benchmark_results:
        seq_lens = []
        speedups = []
        
        for result in results.benchmark_results:
            if 'speedup' in result:
                seq_lens.append(result['seq_len'])
                speedups.append(result['speedup'])
        
        if speedups:
            ax2.plot(seq_lens, speedups, 'bo-', linewidth=2, markersize=8)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Speedup (x)')
            ax2.set_title('Performance Speedup vs Eager Execution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add data labels
            for x, y in zip(seq_lens, speedups):
                ax2.annotate(f'{y:.2f}x', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center')
    
    # 3. Memory Usage
    ax3 = axes[1, 0]
    if results.memory_usage:
        graph_types = list(results.memory_usage.keys())
        memory_mb = [results.memory_usage[gt] / 1024**2 for gt in graph_types
                    if results.memory_usage[gt] > 0]
        valid_graph_types = [gt for gt in graph_types if results.memory_usage[gt] > 0]
        
        if memory_mb:
            bars = ax3.bar(range(len(valid_graph_types)), memory_mb, color='green', alpha=0.7)
            ax3.set_xlabel('Context Length')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('CUDA Graph Memory Usage')
            ax3.set_xticks(range(len(valid_graph_types)))
            ax3.set_xticklabels([f'{gt.name}\n({gt.value})' for gt in valid_graph_types])
            
            # Add value labels
            for bar, mem in zip(bars, memory_mb):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem:.1f}MB', ha='center', va='bottom')
    
    # 4. Execution Time Comparison
    ax4 = axes[1, 1]
    if results.benchmark_results:
        seq_lens = []
        eager_times = []
        graph_times = []
        
        for result in results.benchmark_results:
            seq_lens.append(result['seq_len'])
            eager_times.append(result['eager_avg_ms'])
            
            if 'graph_avg_ms' in result:
                graph_times.append(result['graph_avg_ms'])
            else:
                graph_times.append(None)
        
        # Plot eager times
        ax4.plot(seq_lens, eager_times, 'ro-', linewidth=2, markersize=8, label='Eager')
        
        # Plot graph times where available
        valid_indices = [i for i, t in enumerate(graph_times) if t is not None]
        if valid_indices:
            valid_seq_lens = [seq_lens[i] for i in valid_indices]
            valid_graph_times = [graph_times[i] for i in valid_indices]
            ax4.plot(valid_seq_lens, valid_graph_times, 'go-', linewidth=2,
                    markersize=8, label='CUDA Graph')
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Execution Time (ms)')
        ax4.set_title('Execution Time: Eager vs CUDA Graph')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"cuda_graph_demo_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved as: {filename}")
    
    # Show plot
    plt.show()


def print_summary_report(results: DemoResults):
    """Print a summary report of the demo results."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT - CUDA Graph Compilation (Task 1.3)")
    print("=" * 70)
    
    if not results:
        print("No results to report")
        return
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    
    # Check if graphs were captured successfully
    successful_captures = sum(1 for t in results.graph_capture_times.values() if t > 0)
    total_attempts = len(results.graph_capture_times)
    
    print(f"  Graph Capture Success: {successful_captures}/{total_attempts} context lengths")
    
    # Performance improvement
    speedups = []
    for result in results.benchmark_results:
        if 'speedup' in result:
            speedups.append(result['speedup'])
    
    if speedups:
        avg_speedup = np.mean(speedups)
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.0:
            improvement = (avg_speedup - 1.0) * 100
            print(f"  Performance Improvement: +{improvement:.1f}%")
            
            # Compare to target (10-15%)
            target_min = 10
            target_max = 15
            
            if improvement >= target_min:
                status = "✓ TARGET ACHIEVED"
                if improvement >= target_max:
                    status = "✓ TARGET EXCEEDED"
            else:
                status = "⚠ BELOW TARGET"
                
            print(f"  Target (10-15%): {status}")
    
    # Memory efficiency
    print("\nMEMORY EFFICIENCY:")
    total_memory_mb = sum(m for m in results.memory_usage.values() if m > 0) / 1024**2
    print(f"  Total Graph Memory: {total_memory_mb:.1f} MB")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if speedups:
        avg_speedup = np.mean(speedups)
        if avg_speedup >= 1.10:  # 10% improvement
            print("  1. ✓ CUDA graph compilation is effective for this workload")
            print("  2. Consider implementing for all context length buckets")
            print("  3. Monitor memory usage for large context lengths")
        else:
            print("  1. ⚠ CUDA graph compilation shows limited benefit")
            print("  2. Investigate computation patterns for better graph capture")
            print("  3. Consider alternative optimizations")
    else:
        print("  1. ⚠ No performance data available")
        print("  2. Verify CUDA graph capture is working correctly")
        print("  3. Check system configuration and dependencies")
    
    print("\nNEXT STEPS (Task 1.4):")
    print("  1. Handle multiple context length buckets in production")
    print("  2. Implement dynamic graph selection based on input size")
    print("  3. Add error handling and fallback mechanisms")
    print("  4. Integrate with actual LLM inference pipeline")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Starting CUDA Graph Compilation Demo...")
    print("This demonstrates Phase 1, Task 1.3 implementation")
    print()
    
    try:
        results = run_cuda_graph_demo()
        
        if results:
            print_summary_report(results)
            
            # Save results to file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"task_1_3_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write("CUDA Graph Compilation Report - Task 1.3\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Graph Capture Times:\n")
                for gt, capture_time in results.graph_capture_times.items():
                    f.write(f"  {gt.name}: {capture_time:.3f}s\n")
                
                f.write("\nBenchmark Results:\n")
                for result in results.benchmark_results:
                    f.write(f"\n  seq_len={result['seq_len']}:\n")
                    f.write(f"    Eager: {result['eager_avg_ms']:.2f} ms\n")
                    if 'graph_avg_ms' in result:
                        f.write(f"    Graph: {result['graph_avg_ms']:.2f} ms\n")
                        f.write(f"    Speedup: {result['speedup']:.2f}x\n")
                
                f.write("\nStatistics:\n")
                for key, value in results.statistics.items():
                    f.write(f"  {key}: {value}\n")
            
            print(f"\nDetailed report saved to: {report_file}")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()