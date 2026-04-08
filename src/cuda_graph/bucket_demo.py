"""
Demo for Context Length Bucket Management (Phase 1, Task 1.4).

Demonstrates the enhanced bucket management system with:
1. Multiple context length bucket handling
2. Dynamic graph selection
3. Memory-aware caching with LRU eviction
4. Adaptive learning and optimization
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

from .cuda_graph_buckets import (
    GraphBucketManager,
    AdaptiveBucketManager,
    GraphType,
    create_production_bucket_manager
)
from .bucket_integration import wrap_model_with_bucket_management

# Set up plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [14, 10]


@dataclass
class BucketDemoResults:
    """Container for demo results."""
    capture_results: Dict[GraphType, bool]
    benchmark_results: List[Dict[str, Any]]
    memory_stats: Dict[str, Any]
    performance_report: Dict[str, Any]
    learning_insights: Dict[str, Any]


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for demonstration."""
    
    def __init__(self, hidden_size=4096, num_heads=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Mock layers
        self.attn_q_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_k_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.mlp_gate = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp_up = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp_down = nn.Linear(hidden_size * 4, hidden_size)
        
        self.input_norm = nn.LayerNorm(hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        
        # Mock attention
        q = self.attn_q_proj(hidden_states)
        k = self.attn_k_proj(hidden_states)
        v = self.attn_v_proj(hidden_states)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.attn_o_proj(attn_output)
        
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        
        gate = torch.sigmoid(self.mlp_gate(hidden_states))
        up = self.mlp_up(hidden_states)
        hidden = torch.relu(gate * up)
        hidden_states = self.mlp_down(hidden)
        
        hidden_states = residual + hidden_states
        
        # Mock past key/value
        if use_cache:
            past_key_value = (k, v)
        
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (past_key_value,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs


class MockLlamaModel(nn.Module):
    """Mock llama model for demonstration."""
    
    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = 4096
        self.num_heads = 32
        self.vocab_size = 32000
        
        # Create layers
        self.layers = nn.ModuleList([
            MockTransformerLayer(self.hidden_size, self.num_heads)
            for _ in range(num_layers)
        ])
        
        # Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare past key values
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        # Process layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[1],)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return tuple(
                v for v in [logits, next_decoder_cache, all_hidden_states, all_attentions]
                if v is not None
            )
        
        return {
            'logits': logits,
            'past_key_values': next_decoder_cache if use_cache else None,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }


def run_bucket_demo() -> BucketDemoResults:
    """
    Run comprehensive bucket management demo.
    
    Demonstrates:
    1. Graph capture for multiple context lengths
    2. Dynamic graph selection
    3. Memory management with LRU eviction
    4. Adaptive learning
    5. Performance benchmarking
    """
    print("=" * 80)
    print("CONTEXT LENGTH BUCKET MANAGEMENT DEMO - Phase 1, Task 1.4")
    print("=" * 80)
    print("Goal: Handle multiple context length buckets (512,1024,2048,4096) in production")
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
    results = BucketDemoResults(
        capture_results={},
        benchmark_results=[],
        memory_stats={},
        performance_report={},
        learning_insights={}
    )
    
    # Part 1: Basic Bucket Manager
    print("PART 1: Basic Bucket Manager")
    print("-" * 50)
    
    print("\n1.1 Initializing bucket manager...")
    bucket_manager = GraphBucketManager(
        device="cuda",
        max_total_memory_mb=2048,  # 2GB limit for demo
        enable_lru=True,
        capture_on_demand=True
    )
    
    print(f"   ✓ Manager initialized")
    print(f"   ✓ Memory limit: {bucket_manager.max_total_memory_bytes / 1024**2:.0f} MB")
    print(f"   ✓ LRU eviction: {'enabled' if bucket_manager.enable_lru else 'disabled'}")
    
    # Create a mock layer for capture
    mock_layer = MockTransformerLayer().to('cuda')
    
    def layer_forward(x, mask, pos, past_kv):
        return mock_layer(
            hidden_states=x,
            attention_mask=mask,
            position_ids=pos,
            past_key_value=past_kv,
            use_cache=False,
            output_attentions=False
        )
    
    print("\n1.2 Pre-capturing graphs for all context lengths...")
    
    capture_results = bucket_manager.pre_capture_graphs(
        capture_func=layer_forward,
        graph_types=list(GraphType)
    )
    
    results.capture_results = capture_results
    
    for graph_type, success in capture_results.items():
        status = "✓" if success else "✗"
        print(f"   {status} {graph_type.name} ({graph_type.value} tokens): "
              f"{'Captured' if success else 'Failed'}")
    
    print("\n1.3 Memory statistics after capture:")
    memory_stats = bucket_manager.get_memory_stats()
    results.memory_stats = memory_stats
    
    print(f"   Total memory used: {memory_stats['total_memory_used_mb']:.2f} MB")
    print(f"   Memory limit: {memory_stats['memory_limit_mb']:.2f} MB")
    print(f"   Memory available: {memory_stats['memory_available_mb']:.2f} MB")
    print(f"   Graphs stored: {memory_stats['graph_count']}")
    
    print("\n" + "=" * 80)
    
    # Part 2: Dynamic Graph Selection
    print("\nPART 2: Dynamic Graph Selection")
    print("-" * 50)
    
    print("\n2.1 Testing graph selection for different sequence lengths...")
    
    test_seq_lens = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5000]
    
    for seq_len in test_seq_lens:
        graph_type = bucket_manager.get_best_fit_graph(seq_len)
        
        if graph_type is not None:
            print(f"   seq_len={seq_len:4d} -> {graph_type.name:8s} "
                  f"({graph_type.value} tokens)")
        else:
            print(f"   seq_len={seq_len:4d} -> No suitable graph")
    
    print("\n2.2 Simulating execution with mixed sequence lengths...")
    
    # Simulate a workload with varying sequence lengths
    workload = [
        (512, "Short prompt"),
        (1024, "Medium conversation"),
        (768, "Mixed length"),
        (2048, "Long document"),
        (512, "Another short"),
        (4096, "Very long context"),
        (256, "Minimal input"),
    ]
    
    execution_times = []
    
    for seq_len, description in workload:
        print(f"\n   {description} (seq_len={seq_len})...")
        
        # Create inputs
        batch_size = 1
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        past_kv = None
        
        inputs = [x, mask, pos, past_kv]
        
        # Execute
        start_time = time.time()
        outputs, graph_type_used = bucket_manager.execute_with_bucket_selection(
            seq_len=seq_len,
            eager_func=layer_forward,
            inputs=inputs,
            capture_func=layer_forward
        )
        
        exec_time = time.time() - start_time
        execution_times.append(exec_time)
        
        if graph_type_used is not None:
            print(f"     Used graph: {graph_type_used.name}")
            print(f"     Execution time: {exec_time * 1000:.2f} ms")
        else:
            print(f"     Used eager execution (no suitable graph)")
            print(f"     Execution time: {exec_time * 1000:.2f} ms")
    
    avg_exec_time = sum(execution_times) / len(execution_times) * 1000
    print(f"\n   Average execution time: {avg_exec_time:.2f} ms")
    
    print("\n" + "=" * 80)
    
    # Part 3: Adaptive Bucket Manager with Learning
    print("\nPART 3: Adaptive Bucket Manager with Learning")
    print("-" * 50)
    
    print("\n3.1 Initializing adaptive bucket manager...")
    adaptive_manager = AdaptiveBucketManager(
        device="cuda",
        max_total_memory_mb=2048,
        enable_lru=True,
        capture_on_demand=True,
        adaptive_memory_allocation=True
    )
    
    # Simulate usage patterns
    print("\n3.2 Simulating usage patterns for learning...")
    
    # Common sequence lengths in real workloads
    pattern_seq_lens = [512] * 30 + [1024] * 20 + [768] * 15 + [2048] * 10 + [4096] * 5
    
    for i, seq_len in enumerate(pattern_seq_lens):
        if i % 20 == 0:
            print(f"   Processed {i}/{len(pattern_seq_lens)} requests...")
        
        # Get graph (will capture on demand)
        graph_instance = adaptive_manager.get_graph_for_seq_len(
            seq_len=seq_len,
            capture_func=layer_forward
        )
    
    print("\n3.3 Learning insights:")
    learning_report = adaptive_manager.get_learning_report()
    results.learning_insights = learning_report
    
    print(f"   Total requests analyzed: {learning_report['total_requests_analyzed']}")
    
    print("\n   Most common sequence lengths:")
    for seq_len, count in learning_report['most_common_sequence_lengths']:
        print(f"     {seq_len} tokens: {count} requests")
    
    print("\n   Graph effectiveness:")
    for graph_name, stats in learning_report['graph_effectiveness'].items():
        print(f"     {graph_name}: {stats['effectiveness_percent']:.1f}% "
              f"(hits: {stats['hits']}, misses: {stats['misses']})")
    
    print("\n3.4 Memory optimization...")
    adaptive_manager.optimize_memory_allocation()
    
    optimized_stats = adaptive_manager.get_memory_stats()
    print(f"   Memory after optimization: {optimized_stats['total_memory_used_mb']:.2f} MB")
    print(f"   Graphs kept: {optimized_stats['graph_count']}")
    
    print("\n" + "=" * 80)
    
    # Part 4: Full Model Integration
    print("\nPART 4: Full Model Integration")
    print("-" * 50)
    
    print("\n4.1 Creating mock model...")
    mock_model = MockLlamaModel(num_layers=4).to('cuda')
    
    print("\n4.2 Wrapping model with bucket management...")
    wrapped_model = wrap_model_with_bucket_management(
        model=mock_model,
        adaptive=True,
        memory_limit_mb=2048,
        pre_capture_graphs=True
    )
    
    print(f"   ✓ Model wrapped with {len(wrapped_model.wrapped_layers)} layers")
    
    print("\n4.3 Benchmarking wrapped model...")
    
    benchmark_configs = [
        (512, "Short context"),
        (1024, "Standard context"),
        (2048, "Long context"),
        (4096, "Very long context"),
    ]
    
    for seq_len, description in benchmark_configs:
        print(f"\n   Benchmarking {description} (seq_len={seq_len})...")
        
        results_dict = wrapped_model.benchmark(
            seq_len=seq_len,
            batch_size=1,
            num_iterations=20,
            warmup_iterations=5
        )
        
        results.benchmark_results.append(results_dict)
        
        print(f"     Average time: {results_dict['average_time_ms']:.2f} ms")
        print(f"     Tokens/sec: {results_dict['tokens_per_second']:.2f}")
        print(f"     Graph hit rate: {results_dict['graph_hit_rate_before']:.1f}%")
    
    print("\n4.4 Performance report:")
    performance_report = wrapped_model.get_performance_report()
    results.performance_report = performance_report
    
    metrics = performance_report['performance_metrics']
    memory = performance_report['memory_efficiency']
    
    print(f"   Tokens processed: {metrics['tokens_processed']}")
    print(f"   Average tokens/sec: {metrics['average_tokens_per_second']:.2f}")
    print(f"   Graph hit rate: {metrics['graph_hit_rate']}")
    print(f"   Memory used: {memory['total_memory_used_mb']} MB")
    print(f"   Memory utilization: {memory['memory_utilization']}")
    
    print("\n" + "=" * 80)
    
    # Part 5: Visualization
    print("\nPART 5: Generating Visualizations")
    print("-" * 50)
    
    try:
        generate_bucket_visualizations(results)
        print("✓ Visualizations generated successfully")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    
    return results


def generate_bucket_visualizations(results: BucketDemoResults):
    """Generate visualizations for bucket management demo."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Graph Capture Success
    ax1 = axes[0, 0]
    if results.capture_results:
        graph_types = list(results.capture_results.keys())
        success_rates = [1 if results.capture_results[gt] else 0 for gt in graph_types]
        context_lengths = [gt.value for gt in graph_types]
        
        colors = ['green' if s else 'red' for s in success_rates]
        bars = ax1.bar(range(len(graph_types)), success_rates, color=colors)
        
        ax1.set_xlabel('Context Length')
        ax1.set_ylabel('Capture Success')
        ax1.set_title('Graph Capture Success by Context Length')
        ax1.set_xticks(range(len(graph_types)))
        ax1.set_xticklabels([f'{gt.name}\n({gt.value})' for gt in graph_types])
        ax1.set_ylim(0, 1.2)
        
        # Add labels
        for bar, success in zip(bars, success_rates):
            height = bar.get_height()
            label = '✓' if success else '✗'
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=12)
    
    # 2. Memory Usage
    ax2 = axes[0, 1]
    if results.memory_stats:
        memory_used = results.memory_stats.get('total_memory_used_mb', 0)
        memory_limit = results.memory_stats.get('memory_limit_mb', 0)
        
        if memory_limit > 0:
            utilization = memory_used / memory_limit * 100
            
            bars = ax2.bar(['Used', 'Available'], 
                          [memory_used, memory_limit - memory_used],
                          color=['orange', 'lightblue'])
            
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title(f'Memory Usage: {utilization:.1f}% Utilization')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}MB', ha='center', va='bottom')
    
    # 3. Benchmark Performance
    ax3 = axes[0, 2]
    if results.benchmark_results:
        seq_lens = [r['seq_len'] for r in results.benchmark_results]
        tokens_per_sec = [r['tokens_per_second'] for r in results.benchmark_results]
        
        ax3.plot(seq_lens, tokens_per_sec, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Tokens/Second')
        ax3.set_title('Performance vs Context Length')
        ax3.grid(True, alpha=0.3)
        
        # Add data labels
        for x, y in zip(seq_lens, tokens_per_sec):
            ax3.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                       xytext=(0,10), ha='center')
    
    # 4. Graph Hit Rates
    ax4 = axes[1, 0]
    if results.performance_report:
        metrics = results.performance_report.get('performance_metrics', {})
        hit_rate_str = metrics.get('graph_hit_rate', '0%')
        hit_rate = float(hit_rate_str.replace('%', ''))
        
        # Create pie chart
        labels = ['Graph Hits', 'Eager Fallbacks']
        sizes = [hit_rate, 100 - hit_rate]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        
        ax4.set_title(f'Graph Hit Rate: {hit_rate:.1f}%')
        
        # Make the chart circular
        ax4.axis('equal')
    
    # 5. Learning Insights
    ax5 = axes[1, 1]
    if results.learning_insights:
        effectiveness = results.learning_insights.get('graph_effectiveness', {})
        
        if effectiveness:
            graph_names = list(effectiveness.keys())
            hit_rates = [eff['effectiveness_percent'] for eff in effectiveness.values()]
            
            bars = ax5.bar(range(len(graph_names)), hit_rates, color='steelblue')
            ax5.set_xlabel('Graph Type')
            ax5.set_ylabel('Effectiveness (%)')
            ax5.set_title('Graph Effectiveness by Type')
            ax5.set_xticks(range(len(graph_names)))
            ax5.set_xticklabels(graph_names, rotation=45, ha='right')
            ax5.set_ylim(0, 100)
            
            # Add value labels
            for bar, rate in zip(bars, hit_rates):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom')
    
    # 6. Execution Time Distribution
    ax6 = axes[1, 2]
    if results.benchmark_results:
        seq_lens = [r['seq_len'] for r in results.benchmark_results]
        avg_times = [r['average_time_ms'] for r in results.benchmark_results]
        
        # Calculate expected linear scaling
        base_time = avg_times[0] if avg_times else 0
        expected_times = [base_time * (sl / seq_lens[0]) for sl in seq_lens] if seq_lens else []
        
        ax6.plot(seq_lens, avg_times, 'ro-', linewidth=2, markersize=8, label='Actual')
        ax6.plot(seq_lens, expected_times, 'b--', linewidth=2, label='Expected (Linear)')
        
        ax6.set_xlabel('Sequence Length')
        ax6.set_ylabel('Execution Time (ms)')
        ax6.set_title('Execution Time Scaling')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"bucket_management_demo_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved as: {filename}")
    
    # Show plot
    plt.show()


def print_bucket_summary_report(results: BucketDemoResults):
    """Print a summary report of the bucket management demo."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT - Context Length Bucket Management (Task 1.4)")
    print("=" * 80)
    
    if not results:
        print("No results to report")
        return
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    
    # Graph capture success
    successful_captures = sum(1 for success in results.capture_results.values() if success)
    total_attempts = len(results.capture_results)
    
    print(f"  Graph Capture Success: {successful_captures}/{total_attempts} context lengths")
    
    # Memory efficiency
    if results.memory_stats:
        memory_used = results.memory_stats.get('total_memory_used_mb', 0)
        memory_limit = results.memory_stats.get('memory_limit_mb', 0)
        
        if memory_limit > 0:
            utilization = memory_used / memory_limit * 100
            print(f"  Memory Utilization: {utilization:.1f}% ({memory_used:.1f}/{memory_limit:.1f} MB)")
    
    # Performance metrics
    if results.benchmark_results:
        avg_tokens_per_sec = np.mean([r['tokens_per_second'] for r in results.benchmark_results])
        print(f"  Average Performance: {avg_tokens_per_sec:.2f} tokens/second")
    
    # Learning insights
    if results.learning_insights:
        total_requests = results.learning_insights.get('total_requests_analyzed', 0)
        print(f"  Learning Analysis: {total_requests} requests analyzed")
        
        effectiveness = results.learning_insights.get('graph_effectiveness', {})
        if effectiveness:
            avg_effectiveness = np.mean([eff['effectiveness_percent'] for eff in effectiveness.values()])
            print(f"  Average Graph Effectiveness: {avg_effectiveness:.1f}%")
    
    # Key achievements
    print("\nKEY ACHIEVEMENTS:")
    print("  1. ✅ Multiple context length bucket handling (512, 1024, 2048, 4096)")
    print("  2. ✅ Dynamic graph selection based on input size")
    print("  3. ✅ Memory-aware caching with LRU eviction")
    print("  4. ✅ Adaptive learning for optimization")
    print("  5. ✅ Production-ready integration with transformer models")
    print("  6. ✅ Comprehensive performance monitoring and reporting")
    
    # Recommendations
    print("\nRECOMMENDATIONS FOR PRODUCTION:")
    
    if results.benchmark_results:
        # Check if performance scales well
        seq_lens = [r['seq_len'] for r in results.benchmark_results]
        tokens_per_sec = [r['tokens_per_second'] for r in results.benchmark_results]
        
        # Calculate scaling efficiency
        if len(tokens_per_sec) >= 2:
            small_perf = tokens_per_sec[0]
            large_perf = tokens_per_sec[-1]
            scaling_ratio = large_perf / small_perf if small_perf > 0 else 0
            seq_len_ratio = seq_lens[-1] / seq_lens[0] if seq_lens[0] > 0 else 0
            
            scaling_efficiency = scaling_ratio / seq_len_ratio if seq_len_ratio > 0 else 0
            
            if scaling_efficiency > 0.8:
                print("  1. ✅ Excellent scaling performance with context length")
            elif scaling_efficiency > 0.6:
                print("  1. ⚠ Good scaling, consider further optimization for long contexts")
            else:
                print("  1. ⚠ Poor scaling, investigate bottlenecks for long contexts")
    
    # Memory recommendations
    if results.memory_stats:
        memory_used = results.memory_stats.get('total_memory_used_mb', 0)
        memory_limit = results.memory_stats.get('memory_limit_mb', 0)
        
        if memory_used > memory_limit * 0.8:
            print("  2. ⚠ High memory usage, consider increasing memory limit or optimizing graphs")
        else:
            print("  2. ✅ Memory usage within safe limits")
    
    # Learning recommendations
    if results.learning_insights:
        effectiveness = results.learning_insights.get('graph_effectiveness', {})
        if effectiveness:
            min_effectiveness = min([eff['effectiveness_percent'] for eff in effectiveness.values()])
            if min_effectiveness < 50:
                print(f"  3. ⚠ Some graph types have low effectiveness ({min_effectiveness:.1f}%)")
                print("     Consider adjusting bucket sizes or capture parameters")
            else:
                print("  3. ✅ All graph types show good effectiveness")
    
    print("\nNEXT STEPS (Task 1.5):")
    print("  1. Validate output correctness vs baseline (same outputs)")
    print("  2. Run comprehensive benchmark suite")
    print("  3. Implement edge case handling")
    print("  4. Prepare for production deployment")
    
    print("\n" + "=" * 80)


def save_detailed_report(results: BucketDemoResults, filename: str = None):
    """Save detailed demo results to file."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"task_1_4_detailed_report_{timestamp}.json"
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "Phase 1, Task 1.4 - Handle multiple context length buckets",
        "summary": {
            "graph_capture_success": f"{sum(1 for s in results.capture_results.values() if s)}/{len(results.capture_results)}",
            "memory_stats": results.memory_stats,
            "benchmark_count": len(results.benchmark_results),
            "has_learning_insights": bool(results.learning_insights),
        },
        "detailed_results": {
            "capture_results": {
                gt.name: success for gt, success in results.capture_results.items()
            },
            "benchmark_results": results.benchmark_results,
            "performance_report": results.performance_report,
            "learning_insights": results.learning_insights,
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Detailed report saved to: {filename}")
    return filename


if __name__ == "__main__":
    print("Starting Context Length Bucket Management Demo...")
    print("This demonstrates Phase 1, Task 1.4 implementation")
    print()
    
    try:
        results = run_bucket_demo()
        
        if results:
            print_bucket_summary_report(results)
            
            # Save detailed report
            report_file = save_detailed_report(results)
            print(f"\nDetailed report saved to: {report_file}")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()