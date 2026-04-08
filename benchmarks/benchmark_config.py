#!/usr/bin/env python3
"""
Configuration for benchmarking PipeLLM vs llama.cpp.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""
    name: str
    size: str  # e.g., "7B", "13B", "32B", "70B"
    quantization: str  # e.g., "Q4_0", "Q4_K_M", "Q5_K_M"
    download_url: Optional[str] = None
    local_path: Optional[Path] = None
    context_size: int = 4096
    expected_vram_gb: float = 0.0
    
    def __post_init__(self):
        """Calculate expected VRAM usage."""
        # Rough estimates based on model size and quantization
        size_gb = {
            "7B": 7.0,
            "13B": 13.0,
            "32B": 32.0,
            "70B": 70.0
        }.get(self.size, 0.0)
        
        # Quantization multipliers (approximate)
        quant_mult = {
            "Q4_0": 0.25,
            "Q4_K_M": 0.25,
            "Q5_K_M": 0.31,
            "Q8_0": 0.50,
            "F16": 2.0
        }.get(self.quantization, 0.25)
        
        self.expected_vram_gb = size_gb * quant_mult * 1.1  # 10% overhead

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    model: ModelConfig
    prompt_tokens: int = 512
    generate_tokens: int = 128
    batch_size: int = 1
    threads: int = 8
    gpu_layers: int = -1  # -1 = all layers on GPU
    repeat: int = 5  # Number of times to repeat benchmark
    
    def to_llamacpp_args(self) -> List[str]:
        """Convert to llama.cpp command line arguments."""
        args = [
            "-m", str(self.model.local_path) if self.model.local_path else "",
            "-p", str(self.prompt_tokens),
            "-n", str(self.generate_tokens),
            "-b", str(self.batch_size),
            "-t", str(self.threads),
            "-ngl", str(self.gpu_layers),
            "--no-display-prompt"
        ]
        return [str(arg) for arg in args if arg]

# Benchmark configurations
BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="32B_Q4_short",
        model=ModelConfig(
            name="Llama-2-32B",
            size="32B",
            quantization="Q4_K_M",
            context_size=4096
        ),
        prompt_tokens=256,
        generate_tokens=64
    ),
    BenchmarkConfig(
        name="32B_Q4_standard",
        model=ModelConfig(
            name="Llama-2-32B",
            size="32B",
            quantization="Q4_K_M",
            context_size=4096
        ),
        prompt_tokens=512,
        generate_tokens=128
    ),
    BenchmarkConfig(
        name="70B_Q4_short",
        model=ModelConfig(
            name="Llama-2-70B",
            size="70B",
            quantization="Q4_K_M",
            context_size=4096
        ),
        prompt_tokens=256,
        generate_tokens=64
    ),
]

# Hardware configurations
HARDWARE_CONFIGS = {
    "single_4090": {
        "gpu_name": "RTX 4090",
        "gpu_count": 1,
        "vram_per_gpu_gb": 24,
        "cpu_cores": 16,
        "memory_gb": 64
    },
    "dual_4090": {
        "gpu_name": "RTX 4090",
        "gpu_count": 2,
        "vram_per_gpu_gb": 24,
        "cpu_cores": 16,
        "memory_gb": 64
    }
}

def validate_config(config: BenchmarkConfig, hardware: Dict) -> bool:
    """Validate that benchmark can run on given hardware."""
    required_vram = config.model.expected_vram_gb
    available_vram = hardware["vram_per_gpu_gb"] * hardware["gpu_count"]
    
    if required_vram > available_vram:
        print(f"Warning: Model requires {required_vram:.1f}GB VRAM, "
              f"but only {available_vram}GB available")
        return False
    
    return True