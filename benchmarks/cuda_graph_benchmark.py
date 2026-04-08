#!/usr/bin/env python3
"""
WARNING: SIMULATED DATA — not from real hardware
These numbers are estimates/mocks and must be replaced with real 
measurements before any benchmarks are published (see ROADMAP tasks 
1.6, 2.7, 3.9, 4.2-4.5).

CUDA Graph Benchmark for PipeLLM (Phase 1, Task 1.6).

This module benchmarks the CUDA graph implementation against llama.cpp baseline
and records performance improvements.

IMPORTANT: This benchmark uses simulated timing data and must be
validated with actual CUDA hardware measurements.
"""

import torch
import time
import statistics
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType, create_context_length_buckets
from cuda_graph.llama_integration import wrap_model_with_cuda_graphs
from cuda_graph.output_validation import OutputValidator


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    engine: str
    timings: Dict[str, float]  # mean, stddev, min, max, runs
    tokens_per_sec: Dict[str, float]  # mean, stddev, min, max
    memory_usage_mb: Optional[float] = None
    cuda_graph_info: Optional[Dict[str, Any]] = None
    validation_passed: Optional[bool] = None
    validation_errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "config": self.config_name,
            "engine": self.engine,
            "timings": self.timings,
            "tokens_per_sec": self.tokens_per_sec,
        }
        
        if self.memory_usage_mb is not None:
            result["memory_usage_mb"] = self.memory_usage_mb
        
        if self.cuda_graph_info is not None:
            result["cuda_graph_info"] = self.cuda_graph_info
        
        if self.validation_passed is not None:
            result["validation_passed"] = self.validation_passed
        
        if self.validation_errors is not None:
            result["validation_errors"] = self.validation_errors
        
        return result


class CUDAGraphBenchmark:
    """
    Benchmark CUDA graph implementation against baseline.
    
    This class provides comprehensive benchmarking for Phase 1, Task 1.6,
    measuring performance improvements from CUDA graph compilation.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_validation: bool = True,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10
    ):
        """
        Initialize the CUDA graph benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
            enable_validation: Whether to validate output correctness
            warmup_iterations: Number of warmup iterations before timing
            benchmark_iterations: Number of benchmark iterations
        """
        self.output_dir = output_dir or Path("cuda_graph_benchmark_results")
        self.enable_validation = enable_validation
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CUDA graph manager
        self.graph_manager = CUDAGraphManager()
        
        # Initialize validator if enabled
        self.validator = OutputValidator() if enable_validation else None
        
        # Track benchmark results
        self.results: List[BenchmarkResult] = []
        
        # System info
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "python_version": sys.version,
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
            info["total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        return info
    
    def _create_mock_model_for_benchmark(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        vocab_size: int = 32000
    ):
        """
        Create a mock model for benchmarking.
        
        This creates a simple transformer-like model that can be used
        for benchmarking without requiring a full llama.cpp model.
        """
        import torch.nn as nn
        
        class MockTransformerLayer(nn.Module):
            def __init__(self, hidden_size=4096):
                super().__init__()
                self.hidden_size = hidden_size
                self.linear1 = nn.Linear(hidden_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size)
            
            def forward(self, x, mask=None, pos=None, past_kv=None):
                # Simulate transformer layer computation
                x = self.norm(x)
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return (x,)
        
        class MockLlamaModel(nn.Module):
            def __init__(self, hidden_size=4096, num_layers=4, vocab_size=32000):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.vocab_size = vocab_size
                
                # Create layers
                self.layers = nn.ModuleList([
                    MockTransformerLayer(hidden_size)
                    for _ in range(num_layers)
                ])
                
                # Mock embeddings and head
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
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
                
                # Get embeddings
                hidden_states = self.embed_tokens(input_ids)
                
                # Process through layers
                for layer in self.layers:
                    layer_outputs = layer(
                        hidden_states,
                        mask=attention_mask,
                        pos=position_ids,
                        past_kv=past_key_values
                    )
                    hidden_states = layer_outputs[0]
                
                # Final normalization
                hidden_states = self.norm(hidden_states)
                
                # LM head
                logits = self.lm_head(hidden_states)
                
                if not return_dict:
                    return (logits,)
                
                return {
                    'logits': logits,
                    'past_key_values': None,
                    'hidden_states': None,
                    'attentions': None,
                }
        
        return MockLlamaModel(hidden_size, num_layers, vocab_size)
    
    def _measure_memory_usage(self) -> float:
        """Measure current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        return memory_allocated
    
    def _run_single_iteration(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_graphs: bool = True
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run a single benchmark iteration.
        
        Returns:
            Tuple of (execution_time_ms, output_dict)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Get memory usage
        memory_mb = self._measure_memory_usage()
        
        return execution_time_ms, {
            "outputs": outputs,
            "memory_mb": memory_mb
        }
    
    def benchmark_cuda_graph_vs_eager(
        self,
        config_name: str = "default",
        sequence_lengths: List[int] = None,
        batch_size: int = 1,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_layers: int = 4
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark CUDA graph vs eager execution.
        
        Args:
            config_name: Name of this benchmark configuration
            sequence_lengths: List of sequence lengths to benchmark
            batch_size: Batch size for benchmarking
            hidden_size: Hidden size for model
            vocab_size: Vocabulary size
            num_layers: Number of layers in mock model
            
        Returns:
            Dictionary mapping sequence lengths to benchmark results
        """
        if sequence_lengths is None:
            sequence_lengths = [512, 1024, 2048, 4096]
        
        print(f"\n{'='*80}")
        print(f"Benchmarking CUDA Graph vs Eager Execution")
        print(f"Config: {config_name}")
        print(f"Sequence Lengths: {sequence_lengths}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*80}")
        
        # Create mock model
        print("Creating mock model...")
        model = self._create_mock_model_for_benchmark(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size
        ).to('cuda')
        
        # Wrap model with CUDA graphs
        print("Wrapping model with CUDA graphs...")
        wrapped_model = wrap_model_with_cuda_graphs(
            model=model,
            capture_seq_lens=sequence_lengths
        )
        
        # Pre-capture graphs
        print("Pre-capturing graphs...")
        wrapped_model.capture_all_graphs()
        
        results = {}
        
        for seq_len in sequence_lengths:
            print(f"\nBenchmarking sequence length: {seq_len}")
            
            # Generate test inputs
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len),
                device='cuda', dtype=torch.long
            )
            
            attention_mask = torch.ones(
                batch_size, seq_len,
                device='cuda', dtype=torch.bool
            )
            
            # Benchmark eager execution (graphs disabled)
            print("  Running eager execution (baseline)...")
            wrapped_model.graph_manager.enable_graphs = False
            
            eager_times = []
            eager_memory = []
            
            # Warmup
            for i in range(self.warmup_iterations):
                time_ms, result = self._run_single_iteration(
                    model=wrapped_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_graphs=False
                )
                if i == 0:  # First iteration memory
                    eager_memory.append(result["memory_mb"])
            
            # Benchmark
            for i in range(self.benchmark_iterations):
                time_ms, result = self._run_single_iteration(
                    model=wrapped_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_graphs=False
                )
                eager_times.append(time_ms)
                if i == 0:  # First iteration memory
                    eager_memory.append(result["memory_mb"])
            
            # Benchmark graph execution
            print("  Running CUDA graph execution...")
            wrapped_model.graph_manager.enable_graphs = True
            
            graph_times = []
            graph_memory = []
            
            # Warmup
            for i in range(self.warmup_iterations):
                time_ms, result = self._run_single_iteration(
                    model=wrapped_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_graphs=True
                )
                if i == 0:  # First iteration memory
                    graph_memory.append(result["memory_mb"])
            
            # Benchmark
            for i in range(self.benchmark_iterations):
                time_ms, result = self._run_single_iteration(
                    model=wrapped_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_graphs=True
                )
                graph_times.append(time_ms)
                if i == 0:  # First iteration memory
                    graph_memory.append(result["memory_mb"])
            
            # Calculate statistics
            def calculate_stats(times):
                if not times:
                    return {"mean": 0, "stddev": 0, "min": 0, "max": 0, "runs": 0}
                
                return {
                    "mean": statistics.mean(times),
                    "stddev": statistics.stdev(times) if len(times) > 1 else 0,
                    "min": min(times),
                    "max": max(times),
                    "runs": len(times)
                }
            
            eager_stats = calculate_stats(eager_times)
            graph_stats = calculate_stats(graph_times)
            
            # Calculate tokens per second
            total_tokens = batch_size * seq_len
            
            def calculate_tps(stats):
                if stats["mean"] == 0:
                    return {"mean": 0, "stddev": 0, "min": 0, "max": 0}
                
                mean_tps = total_tokens / (stats["mean"] / 1000)  # Convert ms to seconds
                
                # For stddev, use propagation of uncertainty
                if stats["stddev"] > 0 and stats["mean"] > 0:
                    rel_error = stats["stddev"] / stats["mean"]
                    stddev_tps = mean_tps * rel_error
                else:
                    stddev_tps = 0
                
                min_tps = total_tokens / (stats["max"] / 1000) if stats["max"] > 0 else 0
                max_tps = total_tokens / (stats["min"] / 1000) if stats["min"] > 0 else 0
                
                return {
                    "mean": mean_tps,
                    "stddev": stddev_tps,
                    "min": min_tps,
                    "max": max_tps
                }
            
            eager_tps = calculate_tps(eager_stats)
            graph_tps = calculate_tps(graph_stats)
            
            # Calculate improvement
            if eager_stats["mean"] > 0:
                speedup = eager_stats["mean"] / graph_stats["mean"] if graph_stats["mean"] > 0 else 0
                improvement_pct = (speedup - 1) * 100 if speedup > 0 else 0
            else:
                speedup = 0
                improvement_pct = 0
            
            # Create result objects
            eager_result = BenchmarkResult(
                config_name=f"{config_name}_seq{seq_len}_eager",
                engine="eager",
                timings=eager_stats,
                tokens_per_sec=eager_tps,
                memory_usage_mb=statistics.mean(eager_memory) if eager_memory else None
            )
            
            graph_result = BenchmarkResult(
                config_name=f"{config_name}_seq{seq_len}_cuda_graph",
                engine="cuda_graph",
                timings=graph_stats,
                tokens_per_sec=graph_tps,
                memory_usage_mb=statistics.mean(graph_memory) if graph_memory else None,
                cuda_graph_info={
                    "sequence_length": seq_len,
                    "batch_size": batch_size,
                    "speedup": speedup,
                    "improvement_pct": improvement_pct,
                    "graph_type": wrapped_model.graph_manager.get_best_fit_graph(seq_len).name
                    if wrapped_model.graph_manager.get_best_fit_graph(seq_len) else None
                }
            )
            
            # Run validation if enabled
            if self.enable_validation and self.validator:
                print("  Validating output correctness...")
                
                # Get outputs for validation
                wrapped_model.graph_manager.enable_graphs = False
                eager_outputs = wrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                wrapped_model.graph_manager.enable_graphs = True
                graph_outputs = wrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Compare outputs
                comparison = self.validator._compare_outputs(eager_outputs, graph_outputs)
                
                graph_result.validation_passed = comparison["all_passed"]
                
                if not comparison["all_passed"]:
                    graph_result.validation_errors = [
                        f"Max absolute error: {comparison['max_absolute_error']:.2e}",
                        f"Max relative error: {comparison['max_relative_error']:.2e}"
                    ]
            
            # Store results
            results[seq_len] = {
                "eager": eager_result,
                "cuda_graph": graph_result,
                "improvement": {
                    "speedup": speedup,
                    "improvement_pct": improvement_pct,
                    "eager_mean_ms": eager_stats["mean"],
                    "graph_mean_ms": graph_stats["mean"],
                    "eager_tps": eager_tps["mean"],
                    "graph_tps": graph_tps["mean"]
                }
            }
            
            # Print summary
            print(f"  Results:")
            print(f"    Eager: {eager_stats['mean']:.2f} ms ({eager_tps['mean']:.1f} tokens/sec)")
            print(f"    CUDA Graph: {graph_stats['mean']:.2f} ms ({graph_tps['mean']:.1f} tokens/sec)")
            print(f"    Speedup: {speedup:.2f}x ({improvement_pct:.1f}% improvement)")
            
            if graph_result.validation_passed is not None:
                status = "PASS" if graph_result.validation_passed else "FAIL"
                print(f"    Validation: {status}")
        
        return results
    
    def benchmark_llamacpp_vs_pipellm(
        self,
        config_name: str,
        model_path: Optional[Path] = None,
        prompt_tokens: int = 512,
        generate_tokens: int = 128,
        batch_size: int = 1,
        threads: int = 8,
        gpu_layers: int = -1,
        repeat: int = 5
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark llama.cpp vs PipeLLM with CUDA graphs.
        
        This is a placeholder that will be implemented when PipeLLM
        has full model loading and inference capabilities.
        
        Args:
            config_name: Name of this benchmark configuration
            model_path: Path to model file
            prompt_tokens: Number of prompt tokens
            generate_tokens: Number of tokens to generate
            batch_size: Batch size
            threads: CPU threads to use
            gpu_layers: Number of layers to offload to GPU (-1 = all)
            repeat: Number of times to repeat benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"Benchmarking llama.cpp vs PipeLLM (CUDA Graph)")
        print(f"Config: {config_name}")
        print(f"Model: {model_path}")
        print(f"Prompt: {prompt_tokens} tokens, Generate: {generate_tokens} tokens")
        print(f"{'='*80}")
        
        # For now, return placeholder results
        # This will be implemented when PipeLLM has full model support
        
        print("Note: PipeLLM full model inference not yet implemented.")
        print("Returning placeholder results for benchmarking structure.")
        
        # Placeholder results
        results = {
            "llamacpp": BenchmarkResult(
                config_name=config_name,
                engine="llama.cpp",
                timings={"mean": 100.0, "stddev": 5.0, "min": 95.0, "max": 105.0, "runs": repeat},
                tokens_per_sec={"mean": 25.0, "stddev": 1.2, "min": 23.0, "max": 27.0},
                note="Placeholder - actual llama.cpp benchmark"
            ),
            "pipellm": BenchmarkResult(
                config_name=config_name,
                engine="PipeLLM (CUDA Graph)",
                timings={"mean": 85.0, "stddev": 4.0, "min": 80.0, "max": 90.0, "runs": repeat},
                tokens_per_sec={"mean": 30.0, "stddev": 1.5, "min": 28.0, "max": 32.0},
                cuda_graph_info={
                    "speedup": 1.18,
                    "improvement_pct": 17.6,
                    "note": "Placeholder - expected Phase 1 improvement"
                },
                validation_passed=True,
                note="Placeholder - PipeLLM with CUDA graphs"
            )
        }
        
        # Calculate improvement
        speedup = 100.0 / 85.0  # Placeholder calculation
        improvement_pct = (speedup - 1) * 100
        
        print(f"\nPlaceholder Results:")
        print(f"  llama.cpp: 100.0 ms (25.0 tokens/sec)")
        print(f"  PipeLLM: 85.0 ms (30.0 tokens/sec)")
        print(f"  Speedup: {speedup:.2f}x ({improvement_pct:.1f}% improvement)")
        
        return results
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Handle nested results
                nested = {}
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'to_dict'):
                        nested[subkey] = subvalue.to_dict()
                    else:
                        nested[subkey] = subvalue
                serializable_results[key] = nested
            elif hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        # Add metadata
        data = {
            "metadata": {
                "system_info": self.system_info,
                "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark_config": {
                    "warmup_iterations": self.warmup_iterations,
                    "benchmark_iterations": self.benchmark_iterations,
                    "enable_validation": self.enable_validation
                }
            },
            "results": serializable_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a human-readable summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CUDA GRAPH BENCHMARK SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # System info
        report_lines.append("SYSTEM INFORMATION")
        report_lines.append("-" * 40)
        for key, value in self.system_info.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Benchmark results
        report_lines.append("BENCHMARK RESULTS")
        report_lines.append("-" * 40)
        
        # Check if results are for CUDA graph vs eager
        if any(isinstance(v, dict) and "eager" in v and "cuda_graph" in v for v in results.values()):
            # CUDA graph vs eager results
            total_improvement = 0
            count = 0
            
            for seq_len, seq_results in results.items():
                if isinstance(seq_results, dict) and "improvement" in seq_results:
                    improvement = seq_results["improvement"]
                    
                    report_lines.append(f"Sequence Length: {seq_len}")
                    report_lines.append(f"  Eager: {improvement['eager_mean_ms']:.2f} ms ({improvement['eager_tps']:.1f} tokens/sec)")
                    report_lines.append(f"  CUDA Graph: {improvement['graph_mean_ms']:.2f} ms ({improvement['graph_tps']:.1f} tokens/sec)")
                    report_lines.append(f"  Speedup: {improvement['speedup']:.2f}x ({improvement['improvement_pct']:.1f}% improvement)")
                    report_lines.append("")
                    
                    total_improvement += improvement["improvement_pct"]
                    count += 1
            
            if count > 0:
                avg_improvement = total_improvement / count
                report_lines.append(f"Average Improvement: {avg_improvement:.1f}%")
        
        # Check if results are for llama.cpp vs PipeLLM
        elif "llamacpp" in results and "pipellm" in results:
            # llama.cpp vs PipeLLM results
            llamacpp = results["llamacpp"]
            pipellm = results["pipellm"]
            
            report_lines.append(f"llama.cpp: {llamacpp.timings['mean']:.2f} ms ({llamacpp.tokens_per_sec['mean']:.1f} tokens/sec)")
            report_lines.append(f"PipeLLM: {pipellm.timings['mean']:.2f} ms ({pipellm.tokens_per_sec['mean']:.1f} tokens/sec)")
            
            if hasattr(pipellm, 'cuda_graph_info') and pipellm.cuda_graph_info:
                speedup = pipellm.cuda_graph_info.get("speedup", 0)
                improvement = pipellm.cuda_graph_info.get("improvement_pct", 0)
                report_lines.append(f"Speedup: {speedup:.2f}x ({improvement:.1f}% improvement)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_summary_report(self, results: Dict, filename: str = "benchmark_summary.txt"):
        """Save summary report to file."""
        report = self.generate_summary_report(results)
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {output_path}")
        return output_path


def run_cuda_graph_benchmark():
    """Run the CUDA graph benchmark."""
    print("Starting CUDA Graph Benchmark (Phase 1, Task 1.6)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create benchmark instance
    benchmark = CUDAGraphBenchmark(
        output_dir=Path("benchmark_results/cuda_graph"),
        enable_validation=True,
        warmup_iterations=3,
        benchmark_iterations=10
    )
    
    # Run CUDA graph vs eager benchmark
    print("\n" + "="*80)
    print("Running CUDA Graph vs Eager Benchmark")
    print("="*80)
    
    results = benchmark.benchmark_cuda_graph_vs_eager(
        config_name="phase1_cuda_graph",
        sequence_lengths=[512, 1024, 2048, 4096],
        batch_size=1,
        hidden_size=4096,
        num_layers=4
    )
    
    # Save results
    benchmark.save_results(results, "cuda_graph_vs_eager.json")
    benchmark.save_summary_report(results, "cuda_graph_summary.txt")
    
    # Print summary
    print("\n" + benchmark.generate_summary_report(results))
    
    return benchmark, results


if __name__ == "__main__":
    run_cuda_graph_benchmark()