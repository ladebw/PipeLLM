#!/usr/bin/env python3
"""
CUDA event-based profiler for precise overhead measurement.
This requires CUDA toolkit and PyTorch with CUDA support.
"""

import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class CUDAEventTiming:
    """Timing measurements from CUDA events."""
    kernel_launch_us: float
    kernel_execution_us: float
    memory_copy_us: float
    synchronization_us: float
    total_us: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown."""
        total = self.total_us
        if total == 0:
            return {}
        
        return {
            "kernel_launch": (self.kernel_launch_us / total) * 100,
            "kernel_execution": (self.kernel_execution_us / total) * 100,
            "memory_copy": (self.memory_copy_us / total) * 100,
            "synchronization": (self.synchronization_us / total) * 100
        }

class CUDAMicrobenchmark:
    """Microbenchmarks to measure specific CUDA overheads."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Warm up CUDA
        self._warmup()
    
    def _warmup(self):
        """Warm up CUDA to avoid initialization overhead in measurements."""
        x = torch.randn(1024, 1024, device=self.device)
        y = torch.randn(1024, 1024, device=self.device)
        torch.matmul(x, y)
        torch.cuda.synchronize()
    
    def measure_kernel_launch_overhead(self, size: int = 1024, iterations: int = 1000) -> float:
        """
        Measure kernel launch overhead by timing empty kernel launches.
        
        Args:
            size: Matrix size for dummy operation
            iterations: Number of iterations to average
        
        Returns:
            Average kernel launch overhead in microseconds
        """
        # Create dummy tensors
        x = torch.randn(size, size, device=self.device)
        y = torch.randn(size, size, device=self.device)
        
        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm up
        for _ in range(10):
            torch.matmul(x, y)
        
        torch.cuda.synchronize()
        
        # Measure
        start_event.record()
        for i in range(iterations):
            # Small operation to measure launch overhead
            _ = x + y
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate time per launch
        total_time_ms = start_event.elapsed_time(end_event)
        time_per_launch_us = (total_time_ms * 1000) / iterations
        
        return time_per_launch_us
    
    def measure_memory_copy_overhead(self, size_mb: int = 100, iterations: int = 100) -> float:
        """
        Measure memory copy overhead between CPU and GPU.
        
        Args:
            size_mb: Size of data to copy in MB
            iterations: Number of iterations to average
        
        Returns:
            Average memory copy time in microseconds per MB
        """
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # float32
        
        # Create pinned memory for faster transfers
        cpu_tensor = torch.empty(size_elements, dtype=torch.float32, pin_memory=True)
        gpu_tensor = torch.empty(size_elements, dtype=torch.float32, device=self.device)
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm up
        for _ in range(10):
            gpu_tensor.copy_(cpu_tensor)
        
        torch.cuda.synchronize()
        
        # Measure
        start_event.record()
        for i in range(iterations):
            gpu_tensor.copy_(cpu_tensor)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate time per copy per MB
        total_time_ms = start_event.elapsed_time(end_event)
        time_per_copy_ms = total_time_ms / iterations
        time_per_mb_us = (time_per_copy_ms * 1000) / size_mb
        
        return time_per_mb_us
    
    def measure_synchronization_overhead(self, iterations: int = 1000) -> float:
        """
        Measure CUDA synchronization overhead.
        
        Args:
            iterations: Number of iterations to average
        
        Returns:
            Average synchronization overhead in microseconds
        """
        # Create a small tensor to ensure there's work to synchronize
        x = torch.randn(1024, 1024, device=self.device)
        y = torch.randn(1024, 1024, device=self.device)
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm up
        for _ in range(10):
            torch.matmul(x, y)
            torch.cuda.synchronize()
        
        # Measure synchronization overhead
        start_event.record()
        for i in range(iterations):
            # Small operation
            _ = x + y
            torch.cuda.synchronize()
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate time per synchronization
        total_time_ms = start_event.elapsed_time(end_event)
        time_per_sync_us = (total_time_ms * 1000) / iterations
        
        return time_per_sync_us
    
    def measure_matmul_overhead(self, size: int = 4096, iterations: int = 100) -> float:
        """
        Measure matrix multiplication kernel execution time.
        
        Args:
            size: Matrix size (size x size)
            iterations: Number of iterations to average
        
        Returns:
            Average matmul execution time in microseconds
        """
        x = torch.randn(size, size, device=self.device)
        y = torch.randn(size, size, device=self.device)
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm up
        for _ in range(10):
            torch.matmul(x, y)
        
        torch.cuda.synchronize()
        
        # Measure
        start_event.record()
        for i in range(iterations):
            torch.matmul(x, y)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate time per matmul
        total_time_ms = start_event.elapsed_time(end_event)
        time_per_matmul_us = (total_time_ms * 1000) / iterations
        
        return time_per_matmul_us
    
    def run_comprehensive_benchmark(self) -> Dict:
        """
        Run comprehensive CUDA overhead benchmarks.
        
        Returns:
            Dictionary with all benchmark results
        """
        print("Running comprehensive CUDA overhead benchmarks...")
        
        results = {}
        
        # 1. Kernel launch overhead
        print("  Measuring kernel launch overhead...")
        kernel_launch_us = self.measure_kernel_launch_overhead()
        results["kernel_launch_overhead_us"] = kernel_launch_us
        
        # 2. Memory copy overhead
        print("  Measuring memory copy overhead...")
        memory_copy_us_per_mb = self.measure_memory_copy_overhead()
        results["memory_copy_overhead_us_per_mb"] = memory_copy_us_per_mb
        
        # 3. Synchronization overhead
        print("  Measuring synchronization overhead...")
        sync_overhead_us = self.measure_synchronization_overhead()
        results["synchronization_overhead_us"] = sync_overhead_us
        
        # 4. Matmul execution time (as reference)
        print("  Measuring matmul execution time...")
        matmul_time_us = self.measure_matmul_overhead()
        results["matmul_4096x4096_us"] = matmul_time_us
        
        # 5. Calculate estimated overhead for LLM inference
        print("  Calculating estimated LLM inference overhead...")
        
        # Typical LLM layer parameters for 32B model
        # - Hidden size: 6656
        # - Attention heads: 52
        # - FFN hidden size: 17920
        
        # Estimate per-layer operations
        hidden_size = 6656
        ffn_hidden_size = 17920
        batch_size = 1
        seq_len = 1  # per-token generation
        
        # QKV projection
        qkv_size = hidden_size * 3
        qkv_matmul_us = self._estimate_matmul_time(batch_size * seq_len, hidden_size, qkv_size)
        
        # Attention output projection
        attn_out_matmul_us = self._estimate_matmul_time(batch_size * seq_len, hidden_size, hidden_size)
        
        # FFN up projection
        ffn_up_matmul_us = self._estimate_matmul_time(batch_size * seq_len, hidden_size, ffn_hidden_size)
        
        # FFN down projection
        ffn_down_matmul_us = self._estimate_matmul_time(batch_size * seq_len, ffn_hidden_size, hidden_size)
        
        total_kernel_us = qkv_matmul_us + attn_out_matmul_us + ffn_up_matmul_us + ffn_down_matmul_us
        
        # Estimate overhead
        layers = 60  # Typical for 32B model
        total_layers_us = total_kernel_us * layers
        
        # Launch overhead (one per kernel)
        kernels_per_layer = 4  # QKV, attention out, FFN up, FFN down
        total_launch_us = kernel_launch_us * kernels_per_layer * layers
        
        # Synchronization overhead (one per layer in current llama.cpp)
        total_sync_us = sync_overhead_us * layers
        
        # Memory overhead (estimated)
        # Typical layer weights: ~500MB for 32B Q4
        # Copy per layer: ~8MB (activations + weights for that layer)
        memory_per_layer_mb = 8
        total_memory_us = memory_copy_us_per_mb * memory_per_layer_mb * layers
        
        total_us = total_layers_us + total_launch_us + total_sync_us + total_memory_us
        
        # Breakdown
        overhead_breakdown = {
            "kernel_execution": (total_layers_us / total_us) * 100,
            "kernel_launch": (total_launch_us / total_us) * 100,
            "synchronization": (total_sync_us / total_us) * 100,
            "memory_copy": (total_memory_us / total_us) * 100
        }
        
        results["estimated_llm_overhead"] = {
            "total_time_us": total_us,
            "time_per_token_ms": total_us / 1000,
            "tokens_per_second": 1000000 / total_us,  # 1 second / us per token
            "breakdown_percent": overhead_breakdown,
            "components": {
                "kernel_execution_us": total_layers_us,
                "kernel_launch_us": total_launch_us,
                "synchronization_us": total_sync_us,
                "memory_copy_us": total_memory_us
            }
        }
        
        return results
    
    def _estimate_matmul_time(self, m: int, n: int, k: int) -> float:
        """
        Estimate matmul time based on measured 4096x4096 time.
        Simple linear scaling based on FLOPs.
        """
        # Reference: 4096x4096 matmul
        ref_size = 4096
        ref_time = self.measure_matmul_overhead(size=ref_size, iterations=10)
        
        # FLOPs for matmul: 2 * m * n * k
        ref_flops = 2 * ref_size * ref_size * ref_size
        target_flops = 2 * m * n * k
        
        # Estimate time (not perfectly linear, but good approximation)
        estimated_time = ref_time * (target_flops / ref_flops)
        
        return max(estimated_time, 1.0)  # At least 1us
    
    def save_results(self, results: Dict, output_path: Path):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"CUDA benchmark results saved to: {output_path}")
        
        # Also generate a summary report
        self._generate_summary_report(results, output_path.with_suffix('.md'))
    
    def _generate_summary_report(self, results: Dict, report_path: Path):
        """Generate a summary report."""
        with open(report_path, 'w') as f:
            f.write("# CUDA Overhead Benchmark Results\n\n")
            
            f.write("## Microbenchmark Results\n\n")
            f.write("| Benchmark | Result |\n")
            f.write("|-----------|--------|\n")
            f.write(f"| Kernel Launch Overhead | {results.get('kernel_launch_overhead_us', 0):.2f} µs |\n")
            f.write(f"| Memory Copy Overhead | {results.get('memory_copy_overhead_us_per_mb', 0):.2f} µs/MB |\n")
            f.write(f"| Synchronization Overhead | {results.get('synchronization_overhead_us', 0):.2f} µs |\n")
            f.write(f"| Matmul (4096x4096) | {results.get('matmul_4096x4096_us', 0):.2f} µs |\n")
            
            f.write("\n## Estimated LLM Inference Overhead (32B Model)\n\n")
            
            llm_results = results.get("estimated_llm_overhead", {})
            if llm_results:
                f.write(f"- **Total time per token**: {llm_results.get('time_per_token_ms', 0):.2f} ms\n")
                f.write(f"- **Tokens per second**: {llm_results.get('tokens_per_second', 0):.2f}\n")
                
                f.write("\n### Time Breakdown\n\n")
                f.write("| Component | Percentage | Time (µs) |\n")
                f.write("|-----------|------------|-----------|\n")
                
                breakdown = llm_results.get("breakdown_percent", {})
                components = llm_results.get("components", {})
                
                for comp in ["kernel_execution", "kernel_launch", "synchronization", "memory_copy"]:
                    percent = breakdown.get(comp, 0)
                    time_us = components.get(f"{comp}_us", 0)
                    f.write(f"| {comp.replace('_', ' ').title()} | {percent:.1f}% | {time_us:.0f} |\n")
                
                f.write("\n### Optimization Opportunities\n\n")
                f.write("1. **CUDA Graph Compilation**: Could eliminate kernel launch overhead\n")
                f.write(f"   - Potential improvement: {breakdown.get('kernel_launch', 0):.1f}%\n")
                f.write("2. **Async Memory Transfers**: Could overlap memory copy with compute\n")
                f.write(f"   - Potential improvement: {breakdown.get('memory_copy', 0):.1f}%\n")
                f.write("3. **Pipeline Parallelism**: Could hide synchronization overhead\n")
                f.write(f"   - Potential improvement: {breakdown.get('synchronization', 0):.1f}%\n")
        
        print(f"Summary report saved to: {report_path}")

def main():
    """Main entry point for CUDA profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CUDA overhead microbenchmarks")
    parser.add_argument("--output-dir", default="cuda_benchmarks",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        print("Please run on a system with CUDA-capable GPU")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run benchmarks
    benchmark = CUDAMicrobenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"cuda_benchmarks_{timestamp}.json"
    benchmark.save_results(results, results_file)
    
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()