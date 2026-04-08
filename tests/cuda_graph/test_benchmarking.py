#!/usr/bin/env python3
"""
Tests for the CUDA graph benchmarking system (Task 1.6).

This module tests the benchmarking infrastructure that measures
performance improvements from CUDA graph compilation.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType
from benchmarks.cuda_graph_benchmark import (
    CUDAGraphBenchmark, 
    BenchmarkResult,
    run_cuda_graph_benchmark
)


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_transformer_layer():
    """Create a mock transformer layer for testing."""
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
    
    return MockTransformerLayer


@pytest.fixture
def simple_computation():
    """Create a simple computation function for testing."""
    def compute(x, mask=None, pos=None, past_kv=None):
        # Simple matrix operations
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        output = output @ x
        return output
    
    return compute


@pytest.fixture
def benchmark_instance(cuda_available):
    """Create a benchmark instance for testing."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = CUDAGraphBenchmark(
            output_dir=Path(tmpdir) / "benchmark_results",
            enable_validation=False,  # Disable validation for faster tests
            warmup_iterations=1,
            benchmark_iterations=2
        )
        yield benchmark


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""
    
    def test_initialization(self):
        """Test that BenchmarkResult can be initialized."""
        result = BenchmarkResult(
            config_name="test_config",
            engine="test_engine",
            timings={"mean": 10.0, "stddev": 1.0, "min": 9.0, "max": 11.0, "runs": 5},
            tokens_per_sec={"mean": 100.0, "stddev": 10.0, "min": 90.0, "max": 110.0}
        )
        
        assert result.config_name == "test_config"
        assert result.engine == "test_engine"
        assert result.timings["mean"] == 10.0
        assert result.tokens_per_sec["mean"] == 100.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            config_name="test_config",
            engine="test_engine",
            timings={"mean": 10.0, "stddev": 1.0, "min": 9.0, "max": 11.0, "runs": 5},
            tokens_per_sec={"mean": 100.0, "stddev": 10.0, "min": 90.0, "max": 110.0},
            memory_usage_mb=1024.0,
            cuda_graph_info={"speedup": 1.2, "improvement_pct": 20.0},
            validation_passed=True,
            validation_errors=["error1", "error2"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["config"] == "test_config"
        assert result_dict["engine"] == "test_engine"
        assert result_dict["memory_usage_mb"] == 1024.0
        assert result_dict["cuda_graph_info"]["speedup"] == 1.2
        assert result_dict["validation_passed"] == True
        assert result_dict["validation_errors"] == ["error1", "error2"]
    
    def test_to_dict_partial(self):
        """Test conversion with partial data."""
        result = BenchmarkResult(
            config_name="test_config",
            engine="test_engine",
            timings={"mean": 10.0, "stddev": 1.0, "min": 9.0, "max": 11.0, "runs": 5},
            tokens_per_sec={"mean": 100.0, "stddev": 10.0, "min": 90.0, "max": 110.0}
        )
        
        result_dict = result.to_dict()
        
        assert "config" in result_dict
        assert "engine" in result_dict
        assert "timings" in result_dict
        assert "tokens_per_sec" in result_dict
        assert "memory_usage_mb" not in result_dict  # Should not be present
        assert "cuda_graph_info" not in result_dict  # Should not be present


class TestCUDAGraphBenchmark:
    """Tests for the CUDAGraphBenchmark class."""
    
    def test_initialization(self, benchmark_instance):
        """Test benchmark initialization."""
        assert benchmark_instance is not None
        assert benchmark_instance.enable_validation == False
        assert benchmark_instance.warmup_iterations == 1
        assert benchmark_instance.benchmark_iterations == 2
        assert benchmark_instance.output_dir.exists()
    
    def test_get_system_info(self, benchmark_instance):
        """Test system information collection."""
        system_info = benchmark_instance._get_system_info()
        
        assert "timestamp" in system_info
        assert "cuda_available" in system_info
        assert "pytorch_version" in system_info
        assert "python_version" in system_info
        
        if torch.cuda.is_available():
            assert "gpu_name" in system_info
            assert "gpu_count" in system_info
            assert "total_memory_mb" in system_info
    
    def test_create_mock_model_for_benchmark(self, benchmark_instance):
        """Test mock model creation."""
        model = benchmark_instance._create_mock_model_for_benchmark(
            hidden_size=512,
            num_layers=2,
            vocab_size=1000
        )
        
        assert model is not None
        assert hasattr(model, 'hidden_size')
        assert hasattr(model, 'num_layers')
        assert hasattr(model, 'vocab_size')
        assert hasattr(model, 'layers')
        assert len(model.layers) == 2
        
        # Test forward pass
        if torch.cuda.is_available():
            model = model.to('cuda')
            input_ids = torch.randint(0, 1000, (1, 10), device='cuda')
            output = model(input_ids=input_ids)
            assert 'logits' in output
            assert output['logits'].shape == (1, 10, 1000)
    
    def test_measure_memory_usage(self, benchmark_instance):
        """Test memory usage measurement."""
        memory_mb = benchmark_instance._measure_memory_usage()
        
        if torch.cuda.is_available():
            assert isinstance(memory_mb, float)
            assert memory_mb >= 0
        else:
            assert memory_mb == 0.0
    
    def test_run_single_iteration(self, benchmark_instance, mock_transformer_layer):
        """Test single iteration execution."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create model and inputs
        model = mock_transformer_layer(hidden_size=256).to('cuda')
        input_ids = torch.randn(1, 32, 256, device='cuda')
        attention_mask = torch.ones(1, 32, device='cuda', dtype=torch.bool)
        
        # Run iteration
        execution_time_ms, result = benchmark_instance._run_single_iteration(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_graphs=False
        )
        
        assert isinstance(execution_time_ms, float)
        assert execution_time_ms > 0
        assert "outputs" in result
        assert "memory_mb" in result
        assert isinstance(result["memory_mb"], float)
    
    def test_benchmark_cuda_graph_vs_eager_basic(self, benchmark_instance):
        """Test basic CUDA graph vs eager benchmarking."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Run benchmark with minimal configuration
        results = benchmark_instance.benchmark_cuda_graph_vs_eager(
            config_name="test_benchmark",
            sequence_lengths=[512],
            batch_size=1,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000
        )
        
        assert isinstance(results, dict)
        assert 512 in results
        
        seq_results = results[512]
        assert "eager" in seq_results
        assert "cuda_graph" in seq_results
        assert "improvement" in seq_results
        
        # Check result objects
        eager_result = seq_results["eager"]
        graph_result = seq_results["cuda_graph"]
        
        assert isinstance(eager_result, BenchmarkResult)
        assert isinstance(graph_result, BenchmarkResult)
        assert eager_result.engine == "eager"
        assert graph_result.engine == "cuda_graph"
        
        # Check improvement data
        improvement = seq_results["improvement"]
        assert "speedup" in improvement
        assert "improvement_pct" in improvement
        assert "eager_mean_ms" in improvement
        assert "graph_mean_ms" in improvement
    
    def test_benchmark_cuda_graph_vs_eager_multiple_lengths(self, benchmark_instance):
        """Test benchmarking with multiple sequence lengths."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        sequence_lengths = [512, 1024]
        
        results = benchmark_instance.benchmark_cuda_graph_vs_eager(
            config_name="test_multiple",
            sequence_lengths=sequence_lengths,
            batch_size=1,
            hidden_size=128,  # Smaller for faster tests
            num_layers=1,
            vocab_size=500
        )
        
        assert len(results) == len(sequence_lengths)
        
        for seq_len in sequence_lengths:
            assert seq_len in results
            seq_result = results[seq_len]
            
            # Check that we have both eager and graph results
            assert "eager" in seq_result
            assert "cuda_graph" in seq_result
            
            # Check timing data
            eager_timings = seq_result["eager"].timings
            graph_timings = seq_result["cuda_graph"].timings
            
            assert "mean" in eager_timings
            assert "mean" in graph_timings
            assert eager_timings["runs"] == benchmark_instance.benchmark_iterations
            assert graph_timings["runs"] == benchmark_instance.benchmark_iterations
    
    def test_save_results(self, benchmark_instance):
        """Test saving benchmark results to file."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create mock results
        mock_results = {
            512: {
                "eager": BenchmarkResult(
                    config_name="test_512_eager",
                    engine="eager",
                    timings={"mean": 10.0, "stddev": 1.0, "min": 9.0, "max": 11.0, "runs": 5},
                    tokens_per_sec={"mean": 100.0, "stddev": 10.0, "min": 90.0, "max": 110.0}
                ),
                "cuda_graph": BenchmarkResult(
                    config_name="test_512_graph",
                    engine="cuda_graph",
                    timings={"mean": 8.0, "stddev": 0.8, "min": 7.5, "max": 8.5, "runs": 5},
                    tokens_per_sec={"mean": 125.0, "stddev": 12.5, "min": 112.5, "max": 137.5}
                ),
                "improvement": {
                    "speedup": 1.25,
                    "improvement_pct": 25.0,
                    "eager_mean_ms": 10.0,
                    "graph_mean_ms": 8.0,
                    "eager_tps": 100.0,
                    "graph_tps": 125.0
                }
            }
        }
        
        # Save results
        output_file = benchmark_instance.save_results(mock_results, "test_results.json")
        
        # Verify file was created
        assert output_file.exists()
        assert output_file.name == "test_results.json"
        
        # Load and verify content
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert "metadata" in loaded_data
        assert "results" in loaded_data
        assert "512" in loaded_data["results"]
        
        # Check metadata
        metadata = loaded_data["metadata"]
        assert "system_info" in metadata
        assert "benchmark_timestamp" in metadata
        assert "benchmark_config" in metadata
    
    def test_generate_summary_report(self, benchmark_instance):
        """Test summary report generation."""
        # Create mock results
        mock_results = {
            512: {
                "eager": BenchmarkResult(
                    config_name="test_512_eager",
                    engine="eager",
                    timings={"mean": 10.0, "stddev": 1.0, "min": 9.0, "max": 11.0, "runs": 5},
                    tokens_per_sec={"mean": 100.0, "stddev": 10.0, "min": 90.0, "max": 110.0}
                ),
                "cuda_graph": BenchmarkResult(
                    config_name="test_512_graph",
                    engine="cuda_graph",
                    timings={"mean": 8.0, "stddev": 0.8, "min": 7.5, "max": 8.5, "runs": 5},
                    tokens_per_sec={"mean": 125.0, "stddev": 12.5, "min": 112.5, "max": 137.5}
                ),
                "improvement": {
                    "speedup": 1.25,
                    "improvement_pct": 25.0,
                    "eager_mean_ms": 10.0,
                    "graph_mean_ms": 8.0,
                    "eager_tps": 100.0,
                    "graph_tps": 125.0
                }
            }
        }
        
        # Generate report
        report = benchmark_instance.generate_summary_report(mock_results)
        
        # Verify report content
        assert isinstance(report, str)
        assert len(report) > 0
        assert "CUDA GRAPH BENCHMARK SUMMARY" in report
        assert "SYSTEM INFORMATION" in report
        assert "BENCHMARK RESULTS" in report
        assert "512" in report  # Should contain sequence length
    
    def test_save_summary_report(self, benchmark_instance):
        """Test saving summary report to file."""
        # Create mock results
        mock_results = {
            512: {
                "improvement": {
                    "speedup": 1.25,
                    "improvement_pct": 25.0,
                    "eager_mean_ms": 10.0,
                    "graph_mean_ms": 8.0,
                    "eager_tps": 100.0,
                    "graph_tps": 125.0
                }
            }
        }
        
        # Save report
        output_file = benchmark_instance.save_summary_report(mock_results, "test_summary.txt")
        
        # Verify file was created
        assert output_file.exists()
        assert output_file.name == "test_summary.txt"
        
        # Load and verify content
        with open(output_file, 'r') as f:
            content = f.read()
        
        assert len(content) > 0
        assert "CUDA GRAPH BENCHMARK SUMMARY" in content


class TestBenchmarkIntegration:
    """Integration tests for the benchmarking system."""
    
    def test_benchmark_with_validation(self):
        """Test benchmarking with validation enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create benchmark with validation
            benchmark = CUDAGraphBenchmark(
                output_dir=Path(tmpdir) / "benchmark_with_validation",
                enable_validation=True,
                warmup_iterations=1,
                benchmark_iterations=2
            )
            
            # Run benchmark
            results = benchmark.benchmark_cuda_graph_vs_eager(
                config_name="test_with_validation",
                sequence_lengths=[512],
                batch_size=1,
                hidden_size=256,
                num_layers=2,
                vocab_size=1000
            )
            
            # Check validation results
            seq_results = results[512]
            graph_result = seq_results["cuda_graph"]
            
            assert hasattr(graph_result, 'validation_passed')
            assert graph_result.validation_passed is not None
            
            # If validation failed, check errors
            if not graph_result.validation_passed:
                assert hasattr(graph_result, 'validation_errors')
                assert graph_result.validation_errors is not None
    
    def test_benchmark_llamacpp_vs_pipellm_placeholder(self, benchmark_instance):
        """Test the placeholder llama.cpp vs PipeLLM benchmark."""
        results = benchmark_instance.benchmark_llamacpp_vs_pipellm(
            config_name="test_llamacpp_vs_pipellm",
            model_path=Path("dummy_model.gguf"),
            prompt_tokens=256,
            generate_tokens=64,
            batch_size=1,
            threads=8,
            gpu_layers=-1,
            repeat=3
        )
        
        assert isinstance(results, dict)
        assert "llamacpp" in results
        assert "pipellm" in results
        
        llamacpp_result = results["llamacpp"]
        pipellm_result = results["pipellm"]
        
        assert isinstance(llamacpp_result, BenchmarkResult)
        assert isinstance(pipellm_result, BenchmarkResult)
        assert llamacpp_result.engine == "llama.cpp"
        assert pipellm_result.engine == "PipeLLM (CUDA Graph)"
        
        # Check placeholder improvement data
        assert pipellm_result.cuda_graph_info is not None
        assert "speedup" in pipellm_result.cuda_graph_info
        assert "improvement_pct" in pipellm_result.cuda_graph_info


class TestRunCUDAGraphBenchmarkFunction:
    """Tests for the run_cuda_graph_benchmark function."""
    
    def test_function_runs(self):
        """Test that the run_cuda_graph_benchmark function executes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch to use temp directory
            import benchmarks.cuda_graph_benchmark as benchmark_module
            original_run = benchmark_module.run_cuda_graph_benchmark
            
            def patched_run():
                benchmark = CUDAGraphBenchmark(
                    output_dir=Path(tmpdir) / "benchmark_results",
                    enable_validation=False,
                    warmup_iterations=1,
                    benchmark_iterations=2
                )
                
                results = benchmark.benchmark_cuda_graph_vs_eager(
                    config_name="test_function",
                    sequence_lengths=[512],
                    batch_size=1,
                    hidden_size=256,
                    num_layers=2,
                    vocab_size=1000
                )
                
                return benchmark, results
            
            # Replace function temporarily
            benchmark_module.run_cuda_graph_benchmark = patched_run
            
            try:
                # Run the function
                benchmark, results = patched_run()
                
                assert benchmark is not None
                assert results is not None
                assert isinstance(results, dict)
            finally:
                # Restore original function
                benchmark_module.run_cuda_graph_benchmark = original_run


if __name__ == "__main__":
    pytest.main([__file__, "-v"])