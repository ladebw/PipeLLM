#!/usr/bin/env python3
"""
Setup script for llama.cpp baseline.
Downloads, compiles, and configures llama.cpp for benchmarking.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Output: {result.stdout}")
    return True

def main():
    """Main setup function."""
    # Create benchmarks directory structure
    benchmarks_dir = Path(__file__).parent
    llama_dir = benchmarks_dir / "llama.cpp"
    
    print("Setting up llama.cpp baseline for benchmarking...")
    
    # Clone llama.cpp if not exists
    if not llama_dir.exists():
        print("Cloning llama.cpp repository...")
        if not run_command("git clone https://github.com/ggerganov/llama.cpp.git"):
            return False
    
    # Build llama.cpp
    print("Building llama.cpp...")
    os.chdir(llama_dir)
    
    # Build with CUDA support
    if not run_command("make clean"):
        return False
    
    if not run_command("make LLAMA_CUDA=1 -j4"):
        return False
    
    # Verify build
    if not (llama_dir / "main").exists():
        print("Error: main executable not found after build")
        return False
    
    print("llama.cpp setup complete!")
    print(f"Executable: {llama_dir / 'main'}")
    return True

if __name__ == "__main__":
    if not main():
        sys.exit(1)