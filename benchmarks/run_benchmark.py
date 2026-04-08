#!/usr/bin/env python3
"""
Main benchmarking script for PipeLLM vs llama.cpp.
"""

import json
import subprocess
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from benchmark_config import BENCHMARK_CONFIGS, HARDWARE_CONFIGS, validate_config

class BenchmarkRunner:
    """Runner for benchmarking PipeLLM vs llama.cpp."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hardware": HARDWARE_CONFIGS["single_4090"],  # Default
                "system": self._get_system_info()
            },
            "benchmarks": []
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    
    def run_llamacpp_benchmark(self, config, hardware_config: Dict) -> Dict:
        """Run a benchmark using llama.cpp."""
        print(f"\nRunning llama.cpp benchmark: {config.name}")
        
        # Validate hardware
        if not validate_config(config, hardware_config):
            return None
        
        # Build command
        llama_dir = Path(__file__).parent / "llama.cpp"
        cmd = [str(llama_dir / "main")] + config.to_llamacpp_args()
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run benchmark multiple times
        timings = []
        tokens_per_sec = []
        
        for i in range(config.repeat):
            print(f"  Run {i+1}/{config.repeat}...")
            
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                end_time = time.time()
                
                if result.returncode != 0:
                    print(f"    Error: {result.stderr}")
                    continue
                
                # Parse output for tokens/sec
                for line in result.stdout.split('\n'):
                    if "tokens per second" in line.lower():
                        try:
                            tps = float(line.split(':')[-1].strip())
                            tokens_per_sec.append(tps)
                            break
                        except (ValueError, IndexError):
                            pass
                
                elapsed = end_time - start_time
                timings.append(elapsed)
                print(f"    Time: {elapsed:.2f}s")
                
            except subprocess.TimeoutExpired:
                print("    Timeout after 5 minutes")
                break
        
        if not timings:
            return None
        
        # Calculate statistics
        result = {
            "config": config.name,
            "engine": "llama.cpp",
            "timings": {
                "mean": statistics.mean(timings) if timings else 0,
                "stddev": statistics.stdev(timings) if len(timings) > 1 else 0,
                "min": min(timings) if timings else 0,
                "max": max(timings) if timings else 0,
                "runs": len(timings)
            },
            "tokens_per_sec": {
                "mean": statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
                "stddev": statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0,
                "min": min(tokens_per_sec) if tokens_per_sec else 0,
                "max": max(tokens_per_sec) if tokens_per_sec else 0
            }
        }
        
        print(f"  Results: {result['tokens_per_sec']['mean']:.2f} ± {result['tokens_per_sec']['stddev']:.2f} tokens/sec")
        return result
    
    def run_pipellm_benchmark(self, config, hardware_config: Dict) -> Dict:
        """Run a benchmark using PipeLLM (placeholder for now)."""
        print(f"\nRunning PipeLLM benchmark: {config.name}")
        print("  (PipeLLM engine not yet implemented - using placeholder)")
        
        # For now, return placeholder results
        # In Phase 1, this will be replaced with actual PipeLLM execution
        return {
            "config": config.name,
            "engine": "PipeLLM",
            "timings": {
                "mean": 0,
                "stddev": 0,
                "min": 0,
                "max": 0,
                "runs": 0
            },
            "tokens_per_sec": {
                "mean": 0,
                "stddev": 0,
                "min": 0,
                "max": 0
            },
            "note": "Placeholder - PipeLLM not yet implemented"
        }
    
    def run_all_benchmarks(self, hardware: str = "single_4090"):
        """Run all configured benchmarks."""
        hardware_config = HARDWARE_CONFIGS.get(hardware)
        if not hardware_config:
            print(f"Error: Unknown hardware config '{hardware}'")
            return False
        
        self.results["metadata"]["hardware"] = hardware_config
        
        print(f"Starting benchmarks on {hardware_config['gpu_count']}x {hardware_config['gpu_name']}")
        print(f"Output directory: {self.output_dir}")
        
        for config in BENCHMARK_CONFIGS:
            # Run llama.cpp benchmark
            llama_result = self.run_llamacpp_benchmark(config, hardware_config)
            if llama_result:
                self.results["benchmarks"].append(llama_result)
            
            # Run PipeLLM benchmark
            pipellm_result = self.run_pipellm_benchmark(config, hardware_config)
            if pipellm_result:
                self.results["benchmarks"].append(pipellm_result)
        
        # Save results
        self.save_results()
        return True
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
        
        # Also save a summary CSV
        self.save_summary_csv(filename.with_suffix('.csv'))
    
    def save_summary_csv(self, csv_path: Path):
        """Save summary results to CSV."""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "engine", "tokens_per_sec_mean", 
                "tokens_per_sec_stddev", "time_mean", "time_stddev", "runs"
            ])
            
            for bench in self.results["benchmarks"]:
                writer.writerow([
                    bench["config"],
                    bench["engine"],
                    bench["tokens_per_sec"]["mean"],
                    bench["tokens_per_sec"]["stddev"],
                    bench["timings"]["mean"],
                    bench["timings"]["stddev"],
                    bench["timings"]["runs"]
                ])
        
        print(f"Summary saved to: {csv_path}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PipeLLM vs llama.cpp benchmarks")
    parser.add_argument("--hardware", choices=HARDWARE_CONFIGS.keys(), 
                       default="single_4090", help="Hardware configuration")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(Path(args.output_dir))
    success = runner.run_all_benchmarks(args.hardware)
    
    if success:
        print("\nBenchmarking complete!")
        return 0
    else:
        print("\nBenchmarking failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())