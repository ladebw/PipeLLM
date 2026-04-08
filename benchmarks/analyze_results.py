#!/usr/bin/env python3
"""
Analyze and visualize benchmark results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir: Path = "benchmark_results") -> List[Dict]:
    """Load all benchmark results from directory."""
    results = []
    for file in results_dir.glob("benchmark_results_*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {file}: {e}")
    
    return results

def create_comparison_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame comparing llama.cpp vs PipeLLM results."""
    rows = []
    
    for result in results:
        metadata = result.get("metadata", {})
        hardware = metadata.get("hardware", {})
        
        for bench in result.get("benchmarks", []):
            row = {
                "timestamp": metadata.get("timestamp", ""),
                "hardware": f"{hardware.get('gpu_count', 1)}x {hardware.get('gpu_name', 'GPU')}",
                "config": bench.get("config", ""),
                "engine": bench.get("engine", ""),
                "tokens_per_sec": bench.get("tokens_per_sec", {}).get("mean", 0),
                "tokens_per_sec_stddev": bench.get("tokens_per_sec", {}).get("stddev", 0),
                "time_mean": bench.get("timings", {}).get("mean", 0),
                "time_stddev": bench.get("timings", {}).get("stddev", 0),
                "runs": bench.get("timings", {}).get("runs", 0)
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    sns.set_style("whitegrid")
    
    # Filter out placeholder PipeLLM results (tokens_per_sec = 0)
    df_filtered = df[df["tokens_per_sec"] > 0]
    
    if df_filtered.empty:
        print("No valid benchmark data to plot")
        return
    
    # 1. Bar plot: Tokens/sec by engine and config
    plt.figure(figsize=(12, 6))
    
    # Pivot for grouped bar chart
    pivot_df = df_filtered.pivot_table(
        index="config",
        columns="engine",
        values="tokens_per_sec",
        aggfunc="mean"
    ).fillna(0)
    
    ax = pivot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Tokens per Second: PipeLLM vs llama.cpp")
    plt.ylabel("Tokens per Second")
    plt.xlabel("Benchmark Configuration")
    plt.xticks(rotation=45)
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig(output_dir / "tokens_per_sec_comparison.png", dpi=150)
    plt.close()
    
    # 2. Speedup plot
    if "llama.cpp" in pivot_df.columns and "PipeLLM" in pivot_df.columns:
        plt.figure(figsize=(10, 6))
        speedup = pivot_df["PipeLLM"] / pivot_df["llama.cpp"]
        speedup = speedup[speedup > 0]  # Filter out zeros
        
        if not speedup.empty:
            ax = speedup.plot(kind="bar", color="green", figsize=(10, 6))
            plt.axhline(y=1.0, color="red", linestyle="--", label="Baseline (llama.cpp)")
            plt.title("Speedup: PipeLLM / llama.cpp")
            plt.ylabel("Speedup Factor")
            plt.xlabel("Benchmark Configuration")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "speedup_comparison.png", dpi=150)
            plt.close()
    
    # 3. Time distribution plot
    plt.figure(figsize=(12, 6))
    time_data = df_filtered[["config", "engine", "time_mean", "time_stddev"]].copy()
    time_data["time_min"] = time_data["time_mean"] - time_data["time_stddev"]
    time_data["time_max"] = time_data["time_mean"] + time_data["time_stddev"]
    
    for engine in time_data["engine"].unique():
        engine_data = time_data[time_data["engine"] == engine]
        plt.errorbar(
            engine_data["config"],
            engine_data["time_mean"],
            yerr=engine_data["time_stddev"],
            label=engine,
            marker="o",
            capsize=5
        )
    
    plt.title("Execution Time with Standard Deviation")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Benchmark Configuration")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "execution_time_comparison.png", dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text report of benchmark results."""
    report_path = output_dir / "benchmark_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Benchmark Report: PipeLLM vs llama.cpp\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        
        # Group by engine and config
        grouped = df.groupby(["engine", "config"])
        
        for (engine, config), group in grouped:
            if group["tokens_per_sec"].mean() > 0:  # Skip placeholders
                f.write(f"### {engine} - {config}\n")
                f.write(f"- Tokens/sec: {group['tokens_per_sec'].mean():.2f} ± {group['tokens_per_sec_stddev'].mean():.2f}\n")
                f.write(f"- Time: {group['time_mean'].mean():.2f} ± {group['time_stddev'].mean():.2f} seconds\n")
                f.write(f"- Runs: {group['runs'].sum()}\n\n")
        
        # Comparison table
        f.write("## Performance Comparison\n\n")
        f.write("| Config | llama.cpp (tokens/sec) | PipeLLM (tokens/sec) | Speedup |\n")
        f.write("|--------|------------------------|----------------------|---------|\n")
        
        # Get unique configs
        configs = df["config"].unique()
        
        for config in configs:
            llama_data = df[(df["config"] == config) & (df["engine"] == "llama.cpp")]
            pipellm_data = df[(df["config"] == config) & (df["engine"] == "PipeLLM")]
            
            llama_tps = llama_data["tokens_per_sec"].mean() if not llama_data.empty else 0
            pipellm_tps = pipellm_data["tokens_per_sec"].mean() if not pipellm_data.empty else 0
            
            if llama_tps > 0 and pipellm_tps > 0:
                speedup = pipellm_tps / llama_tps
                f.write(f"| {config} | {llama_tps:.2f} | {pipellm_tps:.2f} | {speedup:.2f}x |\n")
            elif llama_tps > 0:
                f.write(f"| {config} | {llama_tps:.2f} | N/A | N/A |\n")
        
        f.write("\n")
        
        # Hardware info
        f.write("## Hardware Configuration\n\n")
        # Extract from first result
        if not df.empty:
            hardware_info = df.iloc[0]["hardware"]
            f.write(f"- {hardware_info}\n")
        
        f.write("\n## Notes\n\n")
        f.write("- Results are mean ± standard deviation across multiple runs\n")
        f.write("- PipeLLM results are placeholders until engine implementation\n")
        f.write("- Benchmark configurations defined in benchmark_config.py\n")
    
    print(f"Report saved to {report_path}")

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results-dir", default="benchmark_results",
                       help="Directory containing benchmark results")
    parser.add_argument("--output-dir", default="analysis",
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    
    if not results:
        print("No benchmark results found")
        return
    
    print(f"Loaded {len(results)} benchmark result files")
    
    # Create DataFrame
    df = create_comparison_dataframe(results)
    
    # Save raw data
    csv_path = output_dir / "benchmark_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")
    
    # Generate plots
    print("Generating plots...")
    plot_comparison(df, output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(df, output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()