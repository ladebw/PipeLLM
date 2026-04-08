#!/usr/bin/env python3
"""
Tests for cumulative benchmark (Task 2.7).

WARNING: SIMULATED DATA — not from real hardware
These tests verify the benchmark infrastructure, not actual performance.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import benchmark classes
from benchmarks.cumulative_benchmark import (
    CumulativeBenchmark,
    CumulativeBenchmarkResult,
    CumulativeBenchmark as CB
)


class TestCumulativeBenchmarkResult:
    """Tests for CumulativeBenchmarkResult dataclass."""
    
    def test_creation(self):
        """Test creating a CumulativeBenchmarkResult instance."""
        result = CumulativeBenchmarkResult(
            benchmark_name="test_benchmark",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            
            # Performance metrics
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            
            # Improvement percentages
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            
            # Memory usage
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            
            # Validation results
            output_correct=True,
            validation_passed=True,
            validation_details={"test": "data"},
            
            # System info
            system_info={"system": "test"},
            timestamp="2024-01-01T00:00:00"
        )
        
        assert result.benchmark_name == "test_benchmark"
        assert result.sequence_length == 512
        assert result.llama_cpp_tps == 100.0
        assert result.pipellm_async_prefetch_tps == 110.0
        assert result.cumulative_improvement_pct == 10.0
        assert result.output_correct is True
        assert result.validation_passed is True
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = CumulativeBenchmarkResult(
            benchmark_name="test",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["benchmark_name"] == "test"
        assert result_dict["sequence_length"] == 512
        assert result_dict["llama_cpp_tps"] == 100.0
        assert result_dict["cumulative_improvement_pct"] == 10.0
    
    def test_to_json(self):
        """Test converting result to JSON."""
        result = CumulativeBenchmarkResult(
            benchmark_name="test",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        json_str = result.to_json()
        
        assert isinstance(json_str, str)
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["benchmark_name"] == "test"
        assert parsed["cumulative_improvement_pct"] == 10.0


class TestCumulativeBenchmark:
    """Tests for CumulativeBenchmark class."""
    
    def test_initialization(self, tmp_path):
        """Test benchmark initialization."""
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        assert benchmark.output_dir == tmp_path
        assert benchmark.output_dir.exists()
        assert hasattr(benchmark, 'cuda_graph_benchmark')
        assert hasattr(benchmark, 'system_info')
        assert hasattr(benchmark, 'configs')
        
        # Check system info
        assert "system" in benchmark.system_info
        assert "python_version" in benchmark.system_info
        assert "cuda_available" in benchmark.system_info
        assert benchmark.system_info["cuda_available"] is False  # Simulation
    
    def test_get_system_info(self):
        """Test system info collection."""
        benchmark = CumulativeBenchmark()
        system_info = benchmark._get_system_info()
        
        assert isinstance(system_info, dict)
        assert "system" in system_info
        assert "python_version" in system_info
        assert "timestamp" in system_info
        assert "cuda_available" in system_info
        assert "note" in system_info
        assert "WARNING" in system_info["note"]  # Should contain warning
    
    def test_simulate_llama_cpp_benchmark(self):
        """Test llama.cpp benchmark simulation."""
        benchmark = CumulativeBenchmark()
        
        result = benchmark._simulate_llama_cpp_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=5
        )
        
        assert isinstance(result, dict)
        assert "tokens_per_second" in result
        assert "memory_mb" in result
        assert "timing_stats" in result
        assert "note" in result
        assert "SIMULATED" in result["note"]
        
        # Check timing stats
        stats = result["timing_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == 5
    
    def test_simulate_pipellm_eager_benchmark(self):
        """Test PipeLLM eager benchmark simulation."""
        benchmark = CumulativeBenchmark()
        
        result = benchmark._simulate_pipellm_eager_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=5
        )
        
        assert isinstance(result, dict)
        assert "tokens_per_second" in result
        assert "memory_mb" in result
        assert "improvement_vs_llama_pct" in result
        assert "note" in result
        assert "SIMULATED" in result["note"]
        
        # Eager should be slower than llama.cpp
        assert result["improvement_vs_llama_pct"] < 0
    
    def test_simulate_pipellm_cuda_graph_benchmark(self):
        """Test CUDA graph benchmark simulation."""
        benchmark = CumulativeBenchmark()
        
        result = benchmark._simulate_pipellm_cuda_graph_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=5
        )
        
        assert isinstance(result, dict)
        assert "tokens_per_second" in result
        assert "memory_mb" in result
        assert "improvement_vs_eager_pct" in result
        assert "improvement_vs_llama_pct" in result
        assert "note" in result
        assert "SIMULATED" in result["note"]
        
        # CUDA graph should be faster than eager
        assert result["improvement_vs_eager_pct"] > 0
    
    def test_simulate_pipellm_async_prefetch_benchmark(self):
        """Test async prefetch benchmark simulation."""
        benchmark = CumulativeBenchmark()
        
        result = benchmark._simulate_pipellm_async_prefetch_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=5
        )
        
        assert isinstance(result, dict)
        assert "tokens_per_second" in result
        assert "memory_mb" in result
        assert "improvement_vs_cuda_graph_pct" in result
        assert "improvement_vs_llama_pct" in result
        assert "note" in result
        assert "SIMULATED" in result["note"]
        
        # Async prefetch should be faster than CUDA graph
        assert result["improvement_vs_cuda_graph_pct"] > 0
    
    def test_simulate_validation(self):
        """Test validation simulation."""
        benchmark = CumulativeBenchmark()
        
        result = benchmark._simulate_validation(seq_len=512)
        
        assert isinstance(result, dict)
        assert "output_correct" in result
        assert "validation_passed" in result
        assert "max_absolute_error" in result
        assert "max_relative_error" in result
        assert "note" in result
        assert "SIMULATED" in result["note"]
        
        # Validation should pass in simulation
        assert result["output_correct"] is True
        assert result["validation_passed"] is True
    
    @patch.object(CB, '_simulate_llama_cpp_benchmark')
    @patch.object(CB, '_simulate_pipellm_eager_benchmark')
    @patch.object(CB, '_simulate_pipellm_cuda_graph_benchmark')
    @patch.object(CB, '_simulate_pipellm_async_prefetch_benchmark')
    @patch.object(CB, '_simulate_validation')
    def test_run_cumulative_benchmark(
        self,
        mock_validation,
        mock_async,
        mock_cuda_graph,
        mock_eager,
        mock_llama,
        tmp_path
    ):
        """Test running cumulative benchmark with mocked simulations."""
        # Setup mock returns
        mock_llama.return_value = {
            "tokens_per_second": 100.0,
            "memory_mb": 4000.0,
            "timing_stats": {"mean": 10.0, "std": 0.5, "min": 9.5, "max": 10.5, "count": 5}
        }
        
        mock_eager.return_value = {
            "tokens_per_second": 85.0,
            "memory_mb": 4400.0,
            "timing_stats": {"mean": 11.76, "std": 0.6, "min": 11.2, "max": 12.3, "count": 5},
            "improvement_vs_llama_pct": -15.0
        }
        
        mock_cuda_graph.return_value = {
            "tokens_per_second": 95.0,
            "memory_mb": 4620.0,
            "timing_stats": {"mean": 10.53, "std": 0.55, "min": 10.0, "max": 11.1, "count": 5},
            "improvement_vs_eager_pct": 11.76,
            "improvement_vs_llama_pct": -5.0
        }
        
        mock_async.return_value = {
            "tokens_per_second": 110.0,
            "memory_mb": 4989.6,
            "timing_stats": {"mean": 9.09, "std": 0.45, "min": 8.6, "max": 9.6, "count": 5},
            "improvement_vs_cuda_graph_pct": 15.79,
            "improvement_vs_eager_pct": 29.41,
            "improvement_vs_llama_pct": 10.0
        }
        
        mock_validation.return_value = {
            "output_correct": True,
            "validation_passed": True,
            "max_absolute_error": 0.0,
            "max_relative_error": 0.0,
            "num_tensors_compared": 10,
            "num_elements_compared": 512 * 4096,
            "validation_time_ms": 50.0
        }
        
        # Run benchmark
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        result = benchmark.run_cumulative_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=5,
            config_name="test_config"
        )
        
        # Verify result
        assert isinstance(result, CumulativeBenchmarkResult)
        assert result.benchmark_name == "test_config"
        assert result.sequence_length == 512
        assert result.batch_size == 1
        assert result.num_iterations == 5
        
        # Check performance metrics
        assert result.llama_cpp_tps == 100.0
        assert result.pipellm_eager_tps == 85.0
        assert result.pipellm_cuda_graph_tps == 95.0
        assert result.pipellm_async_prefetch_tps == 110.0
        
        # Check improvements (calculated from mock data)
        # CUDA graph vs eager: (95/85 - 1) * 100 = 11.76%
        assert abs(result.cuda_graph_improvement_pct - 11.76) < 0.1
        # Async vs CUDA graph: (110/95 - 1) * 100 = 15.79%
        assert abs(result.async_prefetch_improvement_pct - 15.79) < 0.1
        # Cumulative vs llama.cpp: (110/100 - 1) * 100 = 10.0%
        assert abs(result.cumulative_improvement_pct - 10.0) < 0.1
        
        # Check memory usage
        assert result.llama_cpp_memory_mb == 4000.0
        assert result.pipellm_eager_memory_mb == 4400.0
        assert result.pipellm_cuda_graph_memory_mb == 4620.0
        assert result.pipellm_async_prefetch_memory_mb == 4989.6
        
        # Check validation
        assert result.output_correct is True
        assert result.validation_passed is True
    
    def test_run_all_configurations(self, tmp_path):
        """Test running all configurations."""
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        # Mock the run_cumulative_benchmark method
        mock_result = CumulativeBenchmarkResult(
            benchmark_name="test",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        with patch.object(benchmark, 'run_cumulative_benchmark', return_value=mock_result):
            results = benchmark.run_all_configurations()
        
        # Should have results for all 4 configurations
        assert len(results) == 4
        assert "small_model" in results
        assert "medium_model" in results
        assert "large_model" in results
        assert "xlarge_model" in results
        
        # All results should be CumulativeBenchmarkResult instances
        for result in results.values():
            assert isinstance(result, CumulativeBenchmarkResult)
    
    def test_save_results(self, tmp_path):
        """Test saving benchmark results."""
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        # Create mock results
        mock_result = CumulativeBenchmarkResult(
            benchmark_name="test",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        results = {"test_config": mock_result}
        
        # Save results
        output_path = benchmark.save_results(results)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert "test_config" in saved_data
        assert "_metadata" in saved_data
        assert saved_data["_metadata"]["task"] == "2.7"
        assert "SIMULATED DATA" in saved_data["_metadata"]["warning"]
        
        config_data = saved_data["test_config"]
        assert config_data["benchmark_name"] == "test"
        assert config_data["sequence_length"] == 512
        assert config_data["cumulative_improvement_pct"] == 10.0
    
    def test_generate_summary_report(self, tmp_path):
        """Test generating summary report."""
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        # Create mock results
        mock_result = CumulativeBenchmarkResult(
            benchmark_name="small_model",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        results = {"small_model": mock_result}
        
        # Generate report
        report = benchmark.generate_summary_report(results)
        
        assert isinstance(report, str)
        assert "CUMULATIVE BENCHMARK REPORT" in report
        assert "Phase 1 + Phase 2" in report
        assert "WARNING" in report
        assert "SIMULATED" in report
        assert "small_model" in report
        assert "512" in report
        assert "+10.0%" in report  # Cumulative improvement
    
    def test_save_summary_report(self, tmp_path):
        """Test saving summary report."""
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        # Create mock results
        mock_result = CumulativeBenchmarkResult(
            benchmark_name="test",
            sequence_length=512,
            batch_size=1,
            num_iterations=10,
            llama_cpp_tps=100.0,
            pipellm_eager_tps=85.0,
            pipellm_cuda_graph_tps=95.0,
            pipellm_async_prefetch_tps=110.0,
            cuda_graph_improvement_pct=11.8,
            async_prefetch_improvement_pct=15.8,
            cumulative_improvement_pct=10.0,
            llama_cpp_memory_mb=4000.0,
            pipellm_eager_memory_mb=4400.0,
            pipellm_cuda_graph_memory_mb=4620.0,
            pipellm_async_prefetch_memory_mb=4989.6,
            output_correct=True,
            validation_passed=True,
            validation_details={},
            system_info={},
            timestamp="2024-01-01T00:00:00"
        )
        
        results = {"test_config": mock_result}
        
        # Save report
        output_path = benchmark.save_summary_report(results)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "CUMULATIVE BENCHMARK REPORT" in report_content
        assert "WARNING" in report_content
        assert "SIMULATED" in report_content


class TestIntegration:
    """Integration tests for cumulative benchmark."""
    
    def test_full_benchmark_flow(self, tmp_path):
        """Test full benchmark flow from initialization to result saving."""
        # Initialize benchmark
        benchmark = CumulativeBenchmark(output_dir=tmp_path)
        
        # Run single configuration
        result = benchmark.run_cumulative_benchmark(
            seq_len=512,
            batch_size=1,
            num_iterations=3,  # Small number for test
            config_name="integration_test"
        )
        
        # Verify result
        assert isinstance(result, CumulativeBenchmarkResult)
        assert result.benchmark_name == "integration_test"
        assert result.sequence_length == 512
        
        # Save results
        results = {"integration_test": result}
        json_path = benchmark.save_results(results)
        report_path = benchmark.save_summary_report(results)
        
        # Verify files were created
        assert json_path.exists()
        assert report_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert "integration_test" in json_data
        assert "_metadata" in json_data
        
        # Verify report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = f.read()
        
        assert "integration_test" in report_data
        assert "512" in report_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])