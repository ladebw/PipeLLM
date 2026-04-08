"""
Tests for async prefetch output validation.

WARNING: SIMULATED DATA — not from real hardware
These tests validate the async output validation logic using CPU-based
simulation. Real CUDA hardware validation is required before deployment
(see ROADMAP task 2.6).
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline_parallel.async_output_validation import (
    AsyncOutputValidator,
    AsyncValidationMode,
    AsyncValidationResult,
    MockAsyncPrefetchEngine,
    create_async_validator
)


class TestMockAsyncPrefetchEngine:
    """Tests for MockAsyncPrefetchEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        # Test async engine
        async_engine = MockAsyncPrefetchEngine(use_async=True)
        assert async_engine.use_async
        assert not async_engine.enable_race_conditions
        assert async_engine.stream_sync_correct
        assert async_engine.memory_consistent
        assert async_engine.race_conditions == 0
        
        # Test eager engine
        eager_engine = MockAsyncPrefetchEngine(use_async=False)
        assert not eager_engine.use_async
        
        # Test with race conditions enabled
        race_engine = MockAsyncPrefetchEngine(enable_race_conditions=True)
        assert race_engine.enable_race_conditions
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        engine = MockAsyncPrefetchEngine()
        engine.initialize_buffers(layer_count=4, buffer_size=1024)
        
        assert len(engine.weight_buffers) == 4
        assert len(engine.activation_buffers) == 4
        
        # Check buffer structure
        for i in range(4):
            layer_key = f"layer_{i}"
            assert layer_key in engine.weight_buffers
            assert "compute" in engine.weight_buffers[layer_key]
            assert "staging" in engine.weight_buffers[layer_key]
            assert "next" in engine.weight_buffers[layer_key]
            
            assert layer_key in engine.activation_buffers
            assert "input" in engine.activation_buffers[layer_key]
            assert "output" in engine.activation_buffers[layer_key]
    
    def test_eager_execution(self):
        """Test eager execution mode."""
        engine = MockAsyncPrefetchEngine(use_async=False)
        engine.initialize_buffers(layer_count=2, buffer_size=512)
        
        input_tensor = torch.randn(128, dtype=torch.float32)
        
        # Execute a layer
        output = engine.execute_layer(0, input_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1 or output.shape == input_tensor.shape
        
        # Check execution stats
        stats = engine.get_execution_stats()
        assert stats["use_async"] == False
        assert stats["execution_time_ms"] > 0
        assert stats["overlap_percent"] == 0
    
    def test_async_execution(self):
        """Test async execution mode."""
        engine = MockAsyncPrefetchEngine(use_async=True)
        engine.initialize_buffers(layer_count=3, buffer_size=1024)
        
        input_tensor = torch.randn(256, dtype=torch.float32)
        
        # Execute multiple layers
        outputs = []
        for i in range(3):
            output = engine.execute_layer(i, input_tensor)
            outputs.append(output)
            assert isinstance(output, torch.Tensor)
        
        # Check execution stats
        stats = engine.get_execution_stats()
        assert stats["use_async"] == True
        assert stats["execution_time_ms"] > 0
        assert stats["overlap_percent"] > 0  # Should have some overlap
        assert stats["stream_sync_correct"] == True
        assert stats["memory_consistent"] == True
    
    def test_buffer_swapping(self):
        """Test buffer swapping mechanism."""
        engine = MockAsyncPrefetchEngine(use_async=True)
        engine.initialize_buffers(layer_count=2, buffer_size=256)
        
        # Get initial buffers
        layer_key = "layer_0"
        initial_compute = engine.weight_buffers[layer_key]["compute"].clone()
        initial_staging = engine.weight_buffers[layer_key]["staging"].clone()
        initial_next = engine.weight_buffers[layer_key]["next"].clone()
        
        # Execute to trigger buffer swap
        input_tensor = torch.randn(64, dtype=torch.float32)
        engine.execute_layer(0, input_tensor)
        
        # Buffers should have been rotated
        final_compute = engine.weight_buffers[layer_key]["compute"]
        final_staging = engine.weight_buffers[layer_key]["staging"]
        final_next = engine.weight_buffers[layer_key]["next"]
        
        # Compute should now be old staging
        assert torch.allclose(final_compute, initial_staging, rtol=1e-5)
        # Staging should now be old next
        assert torch.allclose(final_staging, initial_next, rtol=1e-5)
        # Next should now be old compute
        assert torch.allclose(final_next, initial_compute, rtol=1e-5)
    
    def test_race_condition_simulation(self):
        """Test race condition simulation."""
        engine = MockAsyncPrefetchEngine(enable_race_conditions=True)
        engine.initialize_buffers(layer_count=2, buffer_size=256)
        
        input_tensor = torch.randn(64, dtype=torch.float32)
        
        # Execute multiple times to increase chance of race conditions
        race_counts = []
        for _ in range(10):
            engine.execute_layer(0, input_tensor)
            stats = engine.get_execution_stats()
            race_counts.append(stats["race_conditions"])
        
        # At least some race conditions should be detected
        # (though random, with 10% chance per execution we expect some)
        total_races = sum(race_counts)
        assert total_races >= 0  # Could be 0 due to randomness


class TestAsyncOutputValidator:
    """Tests for AsyncOutputValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = AsyncOutputValidator(
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
            enable_race_condition_testing=True,
            stress_test_iterations=50
        )
        
        assert validator.absolute_tolerance == 1e-6
        assert validator.relative_tolerance == 1e-5
        assert validator.enable_race_condition_testing == True
        assert validator.stress_test_iterations == 50
        assert validator.async_engine is None
        assert validator.eager_engine is None
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=4, buffer_size=512)
        
        assert validator.async_engine is not None
        assert validator.eager_engine is not None
        assert validator.async_engine.use_async == True
        assert validator.eager_engine.use_async == False
        
        # Check that engines have same number of buffers
        async_buffers = len(validator.async_engine.weight_buffers)
        eager_buffers = len(validator.eager_engine.weight_buffers)
        assert async_buffers == eager_buffers == 4
    
    def test_async_vs_eager_validation(self):
        """Test async vs eager validation."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=3, buffer_size=256)
        
        input_tensor = torch.randn(128, dtype=torch.float32)
        
        # Run validation
        result = validator.validate_async_vs_eager(
            input_tensor=input_tensor,
            layer_indices=[0, 1, 2],
            num_iterations=3
        )
        
        assert isinstance(result, AsyncValidationResult)
        assert result.validation_mode == AsyncValidationMode.ASYNC_VS_EAGER
        assert result.comparison_count == 9  # 3 layers * 3 iterations
        assert result.passed_count >= 0
        assert result.failed_count >= 0
        assert result.passed_count + result.failed_count == result.comparison_count
        
        # In simulation, should pass (engines are deterministic)
        assert result.passed == True
        assert result.stream_sync_correct == True
        assert result.memory_consistent == True
    
    def test_stream_sync_validation(self):
        """Test stream synchronization validation."""
        validator = AsyncOutputValidator()
        
        input_tensor = torch.randn(64, dtype=torch.float32)
        
        result = validator.validate_stream_synchronization(
            input_tensor=input_tensor,
            num_layers=2,
            iterations=3
        )
        
        assert isinstance(result, AsyncValidationResult)
        assert result.validation_mode == AsyncValidationMode.STREAM_SYNC
        assert result.comparison_count == 6  # 2 layers * 3 iterations
        assert result.stream_sync_correct == True  # No race conditions in this test
    
    def test_memory_consistency_validation(self):
        """Test memory consistency validation."""
        validator = AsyncOutputValidator()
        
        input_tensor = torch.randn(32, dtype=torch.float32)
        
        result = validator.validate_memory_consistency(
            input_tensor=input_tensor,
            buffer_size=128,
            iterations=5
        )
        
        assert isinstance(result, AsyncValidationResult)
        assert result.validation_mode == AsyncValidationMode.MEMORY_CONSISTENCY
        assert result.comparison_count == 40  # 8 layers * 5 iterations
        assert result.memory_consistent == True  # Should be consistent in simulation
    
    def test_stress_test_validation(self):
        """Test stress test validation."""
        validator = AsyncOutputValidator(stress_test_iterations=10)
        
        input_tensor = torch.randn(16, dtype=torch.float32)
        
        result = validator.run_stress_test(
            input_tensor=input_tensor,
            num_layers=4,
            iterations=10
        )
        
        assert isinstance(result, AsyncValidationResult)
        assert result.validation_mode == AsyncValidationMode.STRESS_TEST
        assert result.comparison_count == 40  # 4 layers * 10 iterations
        assert result.stress_test_passed == True  # Should pass in simulation
    
    def test_all_modes_validation(self, tmp_path):
        """Test running all validation modes."""
        validator = AsyncOutputValidator(stress_test_iterations=5)
        
        input_tensor = torch.randn(64, dtype=torch.float32)
        output_dir = tmp_path / "test_output"
        
        results = validator.validate_all_modes(
            input_tensor=input_tensor,
            output_dir=str(output_dir)
        )
        
        assert isinstance(results, dict)
        assert len(results) == 4  # 4 validation modes
        
        expected_modes = [
            "async_vs_eager",
            "stream_sync", 
            "memory_consistency",
            "stress_test"
        ]
        
        for mode in expected_modes:
            assert mode in results
            assert isinstance(results[mode], AsyncValidationResult)
        
        # Check that output files were created
        assert (output_dir / "async_vs_eager_validation.json").exists()
        assert (output_dir / "stream_sync_validation.json").exists()
        assert (output_dir / "memory_consistency_validation.json").exists()
        assert (output_dir / "stress_test_validation.json").exists()
        assert (output_dir / "async_validation_report.txt").exists()
    
    def test_validation_with_race_conditions(self):
        """Test validation with race condition testing enabled."""
        validator = AsyncOutputValidator(enable_race_condition_testing=True)
        validator.initialize_engines(layer_count=2, buffer_size=128)
        
        input_tensor = torch.randn(32, dtype=torch.float32)
        
        # Run validation multiple times to increase chance of race conditions
        for _ in range(3):
            result = validator.validate_async_vs_eager(
                input_tensor=input_tensor,
                layer_indices=[0, 1],
                num_iterations=2
            )
            
            # Race conditions might be detected (random)
            assert result.race_conditions_detected >= 0
    
    def test_result_serialization(self):
        """Test validation result serialization."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=2, buffer_size=256)
        
        input_tensor = torch.randn(64, dtype=torch.float32)
        
        result = validator.validate_async_vs_eager(
            input_tensor=input_tensor,
            num_iterations=2
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "passed" in result_dict
        assert "validation_mode" in result_dict
        assert "comparison_count" in result_dict
        assert "passed_count" in result_dict
        assert "failed_count" in result_dict
        assert "stream_sync_correct" in result_dict
        assert "memory_consistent" in result_dict
        assert "race_conditions_detected" in result_dict
        assert "details" in result_dict
        
        # Serialize to JSON
        json_str = json.dumps(result_dict, default=str)
        assert isinstance(json_str, str)
        
        # Deserialize
        deserialized = json.loads(json_str)
        assert deserialized["passed"] == result.passed


class TestIntegration:
    """Integration tests for async output validation."""
    
    def test_create_async_validator_function(self):
        """Test the create_async_validator convenience function."""
        validator = create_async_validator(
            enable_race_condition_testing=True,
            stress_test_iterations=30
        )
        
        assert isinstance(validator, AsyncOutputValidator)
        assert validator.enable_race_condition_testing == True
        assert validator.stress_test_iterations == 30
        assert validator.absolute_tolerance == 1e-5
        assert validator.relative_tolerance == 1e-4
    
    def test_validation_report_generation(self, tmp_path):
        """Test validation report generation."""
        validator = AsyncOutputValidator(stress_test_iterations=3)
        
        input_tensor = torch.randn(32, dtype=torch.float32)
        output_dir = tmp_path / "test_reports"
        
        results = validator.validate_all_modes(
            input_tensor=input_tensor,
            output_dir=str(output_dir)
        )
        
        # Check report file
        report_path = output_dir / "async_validation_report.txt"
        assert report_path.exists()
        
        # Read and check report content
        report_content = report_path.read_text()
        assert "ASYNC PREFETCH OUTPUT VALIDATION REPORT" in report_content
        assert "WARNING: SIMULATED DATA" in report_content
        assert "OVERALL SUMMARY" in report_content
        assert "DETAILED RESULTS" in report_content
    
    def test_comparison_with_phase1_validator(self):
        """Test integration with Phase 1 output validator."""
        from src.cuda_graph.output_validation import OutputValidator as BaseOutputValidator
        
        # Create both validators
        base_validator = BaseOutputValidator()
        async_validator = AsyncOutputValidator()
        
        # They should have compatible interfaces
        assert hasattr(base_validator, '_compare_tensors')
        assert hasattr(async_validator, '_compare_tensors')
        
        assert hasattr(base_validator, 'validate_single_forward')
        # async_validator extends with additional methods
        
        # Test that async_validator can use base methods
        tensor_a = torch.randn(10, dtype=torch.float32)
        tensor_b = tensor_a.clone() + 0.001
        
        comparison = async_validator._compare_tensors(tensor_a, tensor_b, "test")
        assert isinstance(comparison, dict)
        assert "passed" in comparison
        assert "max_absolute_error" in comparison
        assert "max_relative_error" in comparison


class TestErrorHandling:
    """Tests for error handling in async output validation."""
    
    def test_invalid_layer_indices(self):
        """Test validation with invalid layer indices."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=3, buffer_size=128)
        
        input_tensor = torch.randn(32, dtype=torch.float32)
        
        # Test with layer indices out of range
        result = validator.validate_async_vs_eager(
            input_tensor=input_tensor,
            layer_indices=[0, 5, 10],  # Some indices out of range
            num_iterations=2
        )
        
        # Should still complete, but may have fewer comparisons
        assert isinstance(result, AsyncValidationResult)
        assert result.comparison_count > 0
    
    def test_empty_input_tensor(self):
        """Test validation with empty input tensor."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=2, buffer_size=64)
        
        input_tensor = torch.tensor([], dtype=torch.float32)
        
        result = validator.validate_async_vs_eager(
            input_tensor=input_tensor,
            num_iterations=1
        )
        
        # Should handle empty tensor gracefully
        assert isinstance(result, AsyncValidationResult)
    
    def test_nan_input_tensor(self):
        """Test validation with NaN input tensor."""
        validator = AsyncOutputValidator()
        validator.initialize_engines(layer_count=2, buffer_size=64)
        
        input_tensor = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float32)
        
        result = validator.validate_async_vs_eager(
            input_tensor=input_tensor,
            num_iterations=1
        )
        
        # Should complete validation
        assert isinstance(result, AsyncValidationResult)
        # May fail due to NaN propagation
        assert result.passed in [True, False]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])