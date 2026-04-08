#!/usr/bin/env python3
"""
WARNING: SIMULATED DATA — not from real hardware
These numbers are estimates/mocks and must be replaced with real 
measurements before any benchmarks are published (see ROADMAP tasks 
1.6, 2.7, 3.9, 4.2-4.5).

Profiling tools to measure per-token overhead breakdown in llama.cpp.
Note: This module contains estimated overhead calculations based on
architecture assumptions, not actual hardware measurements.
"""

import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics

@dataclass
class ProfilingResult:
    """Results from profiling a single token generation."""
    total_time_ms: float
    kernel_time_ms: float
    memory_time_ms: float
    dispatch_time_ms: float
    synchronization_time_ms: float
    other_time_ms: float
    tokens_per_second: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown of time spent."""
        total = self.total_time_ms
        if total == 0:
            return {}
        
        return {
            "kernel": (self.kernel_time_ms / total) * 100,
            "memory": (self.memory_time_ms / total) * 100,
            "dispatch": (self.dispatch_time_ms / total) * 100,
            "synchronization": (self.synchronization_time_ms / total) * 100,
            "other": (self.other_time_ms / total) * 100
        }

class LLamaProfiler:
    """Profiler for llama.cpp to measure per-token overhead."""
    
    def __init__(self, llama_path: Path):
        self.llama_path = llama_path
        self.main_executable = llama_path / "main"
        
        if not self.main_executable.exists():
            raise FileNotFoundError(f"llama.cpp main executable not found at {self.main_executable}")
    
    def run_with_timing(self, model_path: Path, prompt_tokens: int = 32, 
                       generate_tokens: int = 1, gpu_layers: int = -1) -> ProfilingResult:
        """
        Run llama.cpp with timing measurements.
        
        Args:
            model_path: Path to GGUF model file
            prompt_tokens: Number of prompt tokens
            generate_tokens: Number of tokens to generate
            gpu_layers: Number of layers to run on GPU (-1 = all)
        
        Returns:
            ProfilingResult with timing breakdown
        """
        # Build command
        cmd = [
            str(self.main_executable),
            "-m", str(model_path),
            "-p", str(prompt_tokens),
            "-n", str(generate_tokens),
            "-ngl", str(gpu_layers),
            "-t", "8",  # 8 threads
            "-b", "1",  # batch size 1
            "--no-display-prompt",
            "--simple-io"  # Simple output for easier parsing
        ]
        
        print(f"Running profiling command: {' '.join(cmd)}")
        
        # Run and measure total time
        start_time = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        end_time = time.perf_counter()
        
        if result.returncode != 0:
            raise RuntimeError(f"llama.cpp failed: {result.stderr}")
        
        total_time_ms = (end_time - start_time) * 1000
        
        # Parse output for tokens/sec
        tokens_per_second = 0
        for line in result.stdout.split('\n'):
            if "tokens per second" in line.lower():
                try:
                    tokens_per_second = float(line.split(':')[-1].strip())
                    break
                except (ValueError, IndexError):
                    pass
        
        # For now, use estimated breakdown based on known llama.cpp architecture
        # In a real implementation, we would use CUDA events or NSight profiling
        breakdown = self._estimate_breakdown(total_time_ms, generate_tokens)
        
        return ProfilingResult(
            total_time_ms=total_time_ms,
            kernel_time_ms=breakdown["kernel"],
            memory_time_ms=breakdown["memory"],
            dispatch_time_ms=breakdown["dispatch"],
            synchronization_time_ms=breakdown["synchronization"],
            other_time_ms=breakdown["other"],
            tokens_per_second=tokens_per_second
        )
    
    def _estimate_breakdown(self, total_time_ms: float, tokens_generated: int) -> Dict[str, float]:
        """
        Estimate time breakdown based on known llama.cpp architecture.
        
        This is an estimation based on:
        - Kernel execution: 40-60% of time (matmul, attention, etc.)
        - Memory transfers: 20-30% of time (weights, activations)
        - Dispatch overhead: 10-15% of time (CPU->GPU launch overhead)
        - Synchronization: 5-10% of time (cudaStreamSynchronize, etc.)
        - Other: 5% of time (Python overhead, I/O, etc.)
        
        These percentages will be refined with actual measurements in later tasks.
        """
        # Base percentages for single token generation
        breakdown = {
            "kernel": total_time_ms * 0.50,      # 50% kernel execution
            "memory": total_time_ms * 0.25,      # 25% memory transfers
            "dispatch": total_time_ms * 0.125,   # 12.5% dispatch overhead
            "synchronization": total_time_ms * 0.075,  # 7.5% synchronization
            "other": total_time_ms * 0.05        # 5% other
        }
        
        # Adjust for multi-token generation (amortized overhead)
        if tokens_generated > 1:
            # Dispatch and synchronization overhead is amortized
            amortization_factor = 1.0 / tokens_generated
            breakdown["dispatch"] *= amortization_factor
            breakdown["synchronization"] *= amortization_factor
            
            # Recalculate other to maintain total
            current_total = sum(breakdown.values())
            breakdown["other"] += (total_time_ms - current_total)
        
        return breakdown
    
    def profile_multiple_runs(self, model_path: Path, runs: int = 10, 
                             prompt_tokens: int = 32, generate_tokens: int = 1) -> List[ProfilingResult]:
        """Run profiling multiple times and collect statistics."""
        results = []
        
        for i in range(runs):
            print(f"Profiling run {i+1}/{runs}...")
            try:
                result = self.run_with_timing(model_path, prompt_tokens, generate_tokens)
                results.append(result)
                print(f"  Time: {result.total_time_ms:.2f}ms, Tokens/sec: {result.tokens_per_second:.2f}")
            except Exception as e:
                print(f"  Run failed: {e}")
        
        return results
    
    def analyze_overhead(self, results: List[ProfilingResult]) -> Dict:
        """Analyze profiling results to understand overhead breakdown."""
        if not results:
            return {}
        
        # Calculate averages
        avg_total = statistics.mean([r.total_time_ms for r in results])
        avg_kernel = statistics.mean([r.kernel_time_ms for r in results])
        avg_memory = statistics.mean([r.memory_time_ms for r in results])
        avg_dispatch = statistics.mean([r.dispatch_time_ms for r in results])
        avg_sync = statistics.mean([r.synchronization_time_ms for r in results])
        avg_other = statistics.mean([r.other_time_ms for r in results])
        avg_tps = statistics.mean([r.tokens_per_second for r in results])
        
        # Calculate standard deviations
        std_total = statistics.stdev([r.total_time_ms for r in results]) if len(results) > 1 else 0
        std_tps = statistics.stdev([r.tokens_per_second for r in results]) if len(results) > 1 else 0
        
        # Create average result
        avg_result = ProfilingResult(
            total_time_ms=avg_total,
            kernel_time_ms=avg_kernel,
            memory_time_ms=avg_memory,
            dispatch_time_ms=avg_dispatch,
            synchronization_time_ms=avg_sync,
            other_time_ms=avg_other,
            tokens_per_second=avg_tps
        )
        
        # Get percentage breakdown
        breakdown = avg_result.get_breakdown()
        
        return {
            "average": avg_result.to_dict(),
            "breakdown_percent": breakdown,
            "statistics": {
                "total_time_ms": {"mean": avg_total, "stddev": std_total},
                "tokens_per_second": {"mean": avg_tps, "stddev": std_tps},
                "runs": len(results)
            },
            "raw_results": [r.to_dict() for r in results]
        }

def save_profiling_results(results: Dict, output_path: Path):
    """Save profiling results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Profiling results saved to: {output_path}")

def generate_profiling_report(results: Dict, output_path: Path):
    """Generate a human-readable profiling report."""
    with open(output_path, 'w') as f:
        f.write("# Per-Token Overhead Breakdown Analysis\n\n")
        
        avg = results.get("average", {})
        breakdown = results.get("breakdown_percent", {})
        stats = results.get("statistics", {})
        
        f.write("## Summary\n\n")
        f.write(f"- Total time per token: {avg.get('total_time_ms', 0):.2f} ± {stats.get('total_time_ms', {}).get('stddev', 0):.2f} ms\n")
        f.write(f"- Tokens per second: {avg.get('tokens_per_second', 0):.2f} ± {stats.get('tokens_per_second', {}).get('stddev', 0):.2f}\n")
        f.write(f"- Number of runs: {stats.get('runs', 0)}\n\n")
        
        f.write("## Time Breakdown (Percentage)\n\n")
        f.write("| Component | Percentage | Time (ms) |\n")
        f.write("|-----------|------------|-----------|\n")
        
        components = [
            ("Kernel Execution", "kernel"),
            ("Memory Transfers", "memory"),
            ("Dispatch Overhead", "dispatch"),
            ("Synchronization", "synchronization"),
            ("Other", "other")
        ]
        
        for name, key in components:
            percent = breakdown.get(key, 0)
            time_ms = avg.get(f"{key}_time_ms", 0)
            f.write(f"| {name} | {percent:.1f}% | {time_ms:.2f} |\n")
        
        f.write("\n")
        
        f.write("## Analysis\n\n")
        f.write("### Key Findings:\n")
        f.write("1. **Dispatch Overhead**: The time spent launching GPU kernels from CPU\n")
        f.write("2. **Synchronization Overhead**: Time waiting for GPU operations to complete\n")
        f.write("3. **Memory Transfer Overhead**: Time spent moving data between CPU and GPU\n")
        f.write("4. **Kernel Execution**: Actual computation time on GPU\n\n")
        
        f.write("### Optimization Opportunities:\n")
        f.write(f"1. **CUDA Graph Compilation**: Could eliminate {breakdown.get('dispatch', 0):.1f}% dispatch overhead\n")
        f.write(f"2. **Async Operations**: Could overlap {breakdown.get('memory', 0):.1f}% memory transfer with compute\n")
        f.write(f"3. **Pipeline Parallelism**: Could hide {breakdown.get('synchronization', 0):.1f}% synchronization overhead\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Validate these estimates with actual CUDA event profiling\n")
        f.write("2. Implement CUDA graph compilation to eliminate dispatch overhead\n")
        f.write("3. Measure actual improvement after optimization\n")
    
    print(f"Profiling report saved to: {output_path}")

def main():
    """Main profiling entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile llama.cpp per-token overhead")
    parser.add_argument("--llama-path", default="benchmarks/llama.cpp",
                       help="Path to llama.cpp directory")
    parser.add_argument("--model-path", required=True,
                       help="Path to GGUF model file")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of profiling runs")
    parser.add_argument("--prompt-tokens", type=int, default=32,
                       help="Number of prompt tokens")
    parser.add_argument("--generate-tokens", type=int, default=1,
                       help="Number of tokens to generate")
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize profiler
    llama_path = Path(args.llama_path)
    profiler = LLamaProfiler(llama_path)
    
    print(f"Starting per-token overhead profiling...")
    print(f"Model: {args.model_path}")
    print(f"Runs: {args.runs}")
    print(f"Prompt tokens: {args.prompt_tokens}")
    print(f"Generate tokens: {args.generate_tokens}")
    print()
    
    # Run profiling
    results = profiler.profile_multiple_runs(
        Path(args.model_path),
        runs=args.runs,
        prompt_tokens=args.prompt_tokens,
        generate_tokens=args.generate_tokens
    )
    
    if not results:
        print("No profiling results collected")
        return
    
    # Analyze results
    analysis = profiler.analyze_overhead(results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"profiling_results_{timestamp}.json"
    report_file = output_dir / f"profiling_report_{timestamp}.md"
    
    save_profiling_results(analysis, results_file)
    generate_profiling_report(analysis, report_file)
    
    print("\nProfiling complete!")
    print(f"Average tokens/sec: {analysis['average']['tokens_per_second']:.2f}")
    print(f"Average time per token: {analysis['average']['total_time_ms']:.2f}ms")

if __name__ == "__main__":
    main()