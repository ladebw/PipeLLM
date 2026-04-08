"""
Tests for Output Validation (Phase 1, Task 1.5).

These tests verify that the output validation system correctly compares
CUDA graph execution outputs with eager execution outputs.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cuda_graph.output_validation import (
    OutputValidator,
    ValidationResult,
    ValidationMode,
    create_default_validator
)
from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType


@pytest.fixture
def cuda_available():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return True


@pytest.fixture
def output_validator(cuda_available):
    """Create an output validator for testing."""
    return OutputValidator(
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        enable_nan_check=True,
        enable_inf_check=True,
        device="cuda"
    )


@pytest.fixture
def simple_computation():
    """Create a simple computation for testing."""
    def compute(x, mask=None, pos=None, past_kv=None):
        # Simple deterministic computation
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        return output
    
    return compute


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
            # Deterministic forward pass
            x = self.norm(x)
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return (x,)
    
    return MockTransformerLayer()


class TestOutputValidator:
    """Test output validator functionality."""
    
    def test_initialization(self, output_validator):
        """Test validator initialization."""
        assert output_validator.absolute_tolerance == 1e-5
        assert output_validator.relative_tolerance == 1e-4
        assert output_validator.enable_nan_check is True
        assert output_validator.enable_inf_check is True
        assert output_validator.device == "cuda"
    
    def test_compare_tensors_identical(self, output_validator):
        """Test tensor comparison with identical tensors."""
        # Create identical tensors
        tensor_a = torch.randn(2, 128, 4096, device='cuda')
        tensor_b = tensor_a.clone()
        
        result = output_validator._compare_tensors(tensor_a, tensor_b, "test_tensor")
        
        assert result["passed"] is True
        assert result["max_absolute_error"] == 0.0
        assert result["max_relative_error"] == 0.0
        assert result["num_elements"] == 2 * 128 * 4096
    
    def test_compare_tensors_slightly_different(self, output_validator):
        """Test tensor comparison with slightly different tensors."""
        tensor_a = torch.randn(2, 128, 4096, device='cuda')
        tensor_b = tensor_a + 1e-6  # Small difference
        
        result = output_validator._compare_tensors(tensor_a, tensor_b, "test_tensor")
        
        # Should pass with our tolerance (1e-5)
        assert result["passed"] is True
        assert result["max_absolute_error"] == 1e-6
        assert result["max_relative_error"] > 0
    
    def test_compare_tensors_very_different(self, output_validator):
        """Test tensor comparison with very different tensors."""
        tensor_a = torch.randn(2, 128, 4096, device='cuda')
        tensor_b = tensor_a + 1.0  # Large difference
        
        result = output_validator._compare_tensors(tensor_a, tensor_b, "test_tensor")
        
        # Should fail with our tolerance (1e-5)
        assert result["passed"] is False
        assert result["max_absolute_error"] == 1.0
        assert result["abs_passed"] is False
    
    def test_compare_tensors_shape_mismatch(self, output_validator):
        """Test tensor comparison with shape mismatch."""
        tensor_a = torch.randn(2, 128, 4096, device='cuda')
        tensor_b = torch.randn(2, 64, 4096, device='cuda')  # Different shape
        
        result = output_validator._compare_tensors(tensor_a, tensor_b, "test_tensor")
        
        assert result["passed"] is False
        assert "Shape mismatch" in result["error"]
    
    def test_compare_tensors_nan_values(self, output_validator):
        """Test tensor comparison with NaN values."""
        tensor_a = torch.randn(2, 128, 4096, device='cuda')
        tensor_b = tensor_a.clone()
        tensor_b[0, 0, 0] = float('nan')  # Introduce NaN
        
        result = output_validator._compare_tensors(tensor_a, tensor_b, "test_tensor")
        
        assert result["passed"] is False
        assert "NaN values detected" in result["error"]
        assert result["b_has_nan"] is True
    
    def test_compare_outputs_simple(self, output_validator):
        """Test comparison of simple output structures."""
        # Single tensor
        output_a = torch.randn(2, 128, 4096, device='cuda')
        output_b = output_a.clone()
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 1
    
    def test_compare_outputs_tuple(self, output_validator):
        """Test comparison of tuple outputs."""
        output_a = (
            torch.randn(2, 128, 4096, device='cuda'),
            torch.randn(2, 128, 4096, device='cuda'),
            None
        )
        output_b = (
            output_a[0].clone(),
            output_a[1].clone(),
            None
        )
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 2
    
    def test_compare_outputs_dict(self, output_validator):
        """Test comparison of dictionary outputs."""
        output_a = {
            'logits': torch.randn(2, 128, 32000, device='cuda'),
            'hidden_states': torch.randn(2, 128, 4096, device='cuda'),
            'attentions': None
        }
        output_b = {
            'logits': output_a['logits'].clone(),
            'hidden_states': output_a['hidden_states'].clone(),
            'attentions': None
        }
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 2
    
    def test_compare_outputs_mixed(self, output_validator):
        """Test comparison of mixed output structures."""
        output_a = {
            'output': (
                torch.randn(2, 128, 4096, device='cuda'),
                [torch.randn(2, 128, 4096, device='cuda')]
            ),
            'metadata': {'batch_size': 2, 'seq_len': 128}
        }
        output_b = {
            'output': (
                output_a['output'][0].clone(),
                [output_a['output'][1][0].clone()]
            ),
            'metadata': {'batch_size': 2, 'seq_len': 128}
        }
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 2


class TestValidationSingleForward:
    """Test single forward pass validation."""
    
    def test_validate_single_forward_identical(self, output_validator, simple_computation):
        """Test validation with identical outputs."""
        # Create test inputs
        batch_size = 1
        seq_len = 128
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = [x, mask, pos, None]
        
        # Use same function for both (should produce identical outputs)
        result = output_validator.validate_single_forward(
            eager_func=simple_computation,
            graph_func=simple_computation,
            inputs=inputs,
            sequence_length=seq_len,
            batch_size=batch_size,
            graph_type="test"
        )
        
        assert result.passed is True
        assert result.max_absolute_error == 0.0
        assert result.max_relative_error == 0.0
        assert result.sequence_length == seq_len
        assert result.batch_size == batch_size
        assert result.graph_type == "test"
        assert result.num_tensors_compared == 1
    
    def test_validate_single_forward_different(self, output_validator):
        """Test validation with different outputs."""
        # Create two different computation functions
        def eager_func(x, mask=None, pos=None, past_kv=None):
            return x * 2.0
        
        def graph_func(x, mask=None, pos=None, past_kv=None):
            return x * 2.0 + 0.1  # Different output
        
        # Create test inputs
        x = torch.randn(1, 64, 4096, device='cuda')
        inputs = [x, None, None, None]
        
        result = output_validator.validate_single_forward(
            eager_func=eager_func,
            graph_func=graph_func,
            inputs=inputs,
            sequence_length=64,
            batch_size=1
        )
        
        assert result.passed is False
        assert result.max_absolute_error == 0.1
        assert result.error_details is not None


class TestValidationMultipleContextLengths:
    """Test validation across multiple context lengths."""
    
    def test_validate_multiple_context_lengths(self, output_validator, simple_computation):
        """Test validation across different context lengths."""
        # Use same function for both (should pass)
        context_lengths = [64, 128, 256]  # Small lengths for quick testing
        
        results = output_validator.validate_multiple_context_lengths(
            eager_func=simple_computation,
            graph_func=simple_computation,
            context_lengths=context_lengths,
            batch_size=1,
            hidden_size=1024,  # Smaller for quick testing
            num_iterations=2
        )
        
        assert len(results) == len(context_lengths)
        
        for seq_len in context_lengths:
            assert seq_len in results
            result = results[seq_len]
            assert result.passed is True
            assert result.sequence_length == seq_len
    
    def test_validate_multiple_context_lengths_with_failure(self, output_validator):
        """Test validation with a failing computation."""
        def eager_func(x, mask=None, pos=None, past_kv=None):
            return x
        
        def graph_func(x, mask=None, pos=None, past_kv=None):
            # Introduce error for certain sequence lengths
            seq_len = x.shape[1]
            if seq_len == 128:
                return x + 0.01  # Error for seq_len=128
            return x
        
        context_lengths = [64, 128, 256]
        
        results = output_validator.validate_multiple_context_lengths(
            eager_func=eager_func,
            graph_func=graph_func,
            context_lengths=context_lengths,
            batch_size=1,
            hidden_size=1024,
            num_iterations=1
        )
        
        # Check results
        assert results[64].passed is True
        assert results[128].passed is False  # Should fail
        assert results[256].passed is True


class TestValidationWithCUDAGraph:
    """Test validation with actual CUDA graph capture."""
    
    def test_validate_cuda_graph_vs_eager(self, output_validator, simple_computation):
        """Test validation comparing CUDA graph vs eager execution."""
        # Create graph manager
        graph_manager = CUDAGraphManager()
        
        # Capture a graph
        graph_manager.capture_graph(
            graph_type=GraphType.STANDARD,
            capture_func=simple_computation
        )
        
        # Create test inputs
        batch_size = 1
        seq_len = 1024
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = [x, mask, pos, None]
        
        # Define eager function
        def eager_func(*args):
            return simple_computation(*args)
        
        # Define graph function
        def graph_func(*args):
            return graph_manager.execute_graph(GraphType.STANDARD, args)
        
        # Validate
        result = output_validator.validate_single_forward(
            eager_func=eager_func,
            graph_func=graph_func,
            inputs=inputs,
            sequence_length=seq_len,
            batch_size=batch_size,
            graph_type="STANDARD"
        )
        
        # CUDA graph should produce identical outputs
        assert result.passed is True
        assert result.graph_type == "STANDARD"


class TestValidationReport:
    """Test validation report generation."""
    
    def test_generate_validation_report(self, output_validator):
        """Test report generation."""
        # Create mock results
        results = {
            512: ValidationResult(
                passed=True,
                max_absolute_error=1e-6,
                max_relative_error=1e-5,
                mean_absolute_error=5e-7,
                mean_relative_error=5e-6,
                num_tensors_compared=3,
                num_elements_compared=1000000,
                validation_time_ms=150.5,
                sequence_length=512,
                batch_size=1,
                graph_type="SHORT"
            ),
            1024: ValidationResult(
                passed=False,
                max_absolute_error=0.1,
                max_relative_error=0.05,
                mean_absolute_error=0.01,
                mean_relative_error=0.005,
                num_tensors_compared=3,
                num_elements_compared=2000000,
                validation_time_ms=300.2,
                sequence_length=1024,
                batch_size=1,
                graph_type="STANDARD",
                warnings=["High absolute error: 1.00e-01"]
            )
        }
        
        report = output_validator.generate_validation_report(results)
        
        # Check report contains expected information
        assert "CUDA Graph Output Validation Report" in report
        assert "Total tests: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "Context Length: 512" in report
        assert "Context Length: 1024" in report
        assert "Warnings:" in report
    
    def test_save_validation_results(self, output_validator):
        """Test saving validation results to disk."""
        # Create mock results
        results = {
            512: ValidationResult(
                passed=True,
                max_absolute_error=1e-6,
                max_relative_error=1e-5,
                mean_absolute_error=5e-7,
                mean_relative_error=5e-6,
                num_tensors_compared=3,
                num_elements_compared=1000000,
                validation_time_ms=150.5,
                sequence_length=512,
                batch_size=1,
                graph_type="SHORT"
            )
        }
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "validation_results"
            
            output_validator.save_validation_results(
                validation_results=results,
                output_dir=output_dir,
                filename="test_results.json"
            )
            
            # Check files were created
            json_path = output_dir / "test_results.json"
            report_path = output_dir / "validation_report.txt"
            
            assert json_path.exists()
            assert report_path.exists()
            
            # Check JSON content
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "results" in data
            assert "512" in data["results"]
            assert data["results"]["512"]["passed"] is True


class TestDefaultValidator:
    """Test default validator creation."""
    
    def test_create_default_validator(self, cuda_available):
        """Test creating default validator."""
        validator = create_default_validator()
        
        assert isinstance(validator, OutputValidator)
        assert validator.absolute_tolerance == 1e-5
        assert validator.relative_tolerance == 1e-4
        assert validator.enable_nan_check is True
        assert validator.enable_inf_check is True
        assert validator.device == "cuda"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_outputs(self, output_validator):
        """Test validation with empty outputs."""
        result = output_validator._compare_outputs([], [])
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 0
        assert result["num_elements_compared"] == 0
    
    def test_none_outputs(self, output_validator):
        """Test validation with None outputs."""
        result = output_validator._compare_outputs(None, None)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 0
    
    def test_mixed_none_tensors(self, output_validator):
        """Test validation with mixed None and tensors."""
        output_a = (torch.randn(2, 128, 4096, device='cuda'), None)
        output_b = (output_a[0].clone(), None)
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is True
        assert result["num_tensors_compared"] == 1
    
    def test_different_structures(self, output_validator):
        """Test validation with different output structures."""
        output_a = torch.randn(2, 128, 4096, device='cuda')
        output_b = [output_a.clone()]  # Different structure
        
        result = output_validator._compare_outputs(output_a, output_b)
        
        assert result["all_passed"] is False


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])