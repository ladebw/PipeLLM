"""
Tests for CUDA Graph Capture implementation.

These tests verify the correctness and performance of CUDA graph
capture for LLM decode loops (Phase 1, Task 1.3).
"""

import pytest
import torch
import sys
import os
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cuda_graph.cuda_graph_capture import (
    CUDAGraphManager,
    GraphType,
    GraphInstance,
    benchmark_graph_vs_eager,
    create_context_length_buckets
)


@pytest.fixture
def cuda_available():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return True


@pytest.fixture
def graph_manager(cuda_available):
    """Create a CUDA graph manager for testing."""
    return CUDAGraphManager()


@pytest.fixture
def simple_computation():
    """Create a simple computation for testing."""
    def compute(x, y):
        # Simple matrix operations
        z = x @ y.T
        z = torch.relu(z)
        z = z.mean(dim=1, keepdim=True)
        return z
    return compute


class TestCUDAGraphManager:
    """Test CUDA Graph Manager functionality."""
    
    def test_initialization(self, graph_manager):
        """Test that graph manager initializes correctly."""
        assert graph_manager.device.type == 'cuda'
        assert len(graph_manager.graphs) == 0
        assert graph_manager.stream is not None
        assert not graph_manager._is_capturing
        
        # Check statistics
        stats = graph_manager.get_stats()
        assert stats["graph_executions"] == 0
        assert stats["eager_executions"] == 0
        assert stats["graph_hit_rate"] == 0.0
    
    def test_create_sample_inputs(self, graph_manager):
        """Test sample input creation."""
        batch_size = 2
        seq_len = 128
        
        inputs = graph_manager._create_sample_inputs(batch_size, seq_len)
        
        # Should return 4 items: input_ids, attention_mask, position_ids, past_key_values
        assert len(inputs) == 4
        
        # Check input_ids
        input_ids = inputs[0]
        assert input_ids.shape == (batch_size, seq_len)
        assert input_ids.device.type == 'cuda'
        
        # Check attention_mask
        attention_mask = inputs[1]
        assert attention_mask.shape == (batch_size, seq_len)
        assert attention_mask.dtype == torch.bool
        
        # Check position_ids
        position_ids = inputs[2]
        assert position_ids.shape == (batch_size, seq_len)
        
        # Check past_key_values
        past_key_values = inputs[3]
        assert isinstance(past_key_values, list)
        assert len(past_key_values) == 32  # 32 layers for 32B model
        
        for key, value in past_key_values:
            assert key.shape == (batch_size, 32, seq_len - 1, 128)  # 32 heads, head_dim=128
            assert value.shape == (batch_size, 32, seq_len - 1, 128)
    
    def test_make_static_tensors(self, graph_manager):
        """Test static tensor creation."""
        # Create regular tensors
        regular_tensors = [
            torch.randn(2, 128, device='cuda'),
            torch.ones(2, 128, dtype=torch.bool, device='cuda'),
        ]
        
        static_tensors = graph_manager._make_static_tensors(regular_tensors)
        
        assert len(static_tensors) == len(regular_tensors)
        
        for static, regular in zip(static_tensors, regular_tensors):
            assert static.shape == regular.shape
            assert static.dtype == regular.dtype
            assert static.device == regular.device
            # Static tensors should be empty (uninitialized)
            assert torch.all(static == 0) or torch.all(~static)  # For bool tensors
    
    def test_get_best_fit_graph(self, graph_manager):
        """Test graph selection logic."""
        # Capture some graphs
        graph_types = [GraphType.SHORT, GraphType.STANDARD, GraphType.LONG]
        
        # Test with no graphs captured
        assert graph_manager.get_best_fit_graph(256) is None
        assert graph_manager.get_best_fit_graph(1024) is None
        
        # Simulate having graphs captured
        for gt in graph_types:
            graph_manager.graphs[gt] = None  # Dummy placeholder
        
        # Test graph selection
        assert graph_manager.get_best_fit_graph(256) == GraphType.SHORT
        assert graph_manager.get_best_fit_graph(512) == GraphType.SHORT
        assert graph_manager.get_best_fit_graph(600) == GraphType.STANDARD
        assert graph_manager.get_best_fit_graph(1024) == GraphType.STANDARD
        assert graph_manager.get_best_fit_graph(3000) == GraphType.LONG
        assert graph_manager.get_best_fit_graph(4096) == GraphType.LONG
        assert graph_manager.get_best_fit_graph(5000) is None  # Too large
    
    def test_clear_graphs(self, graph_manager):
        """Test graph clearing."""
        # Add dummy graphs
        graph_manager.graphs[GraphType.SHORT] = "dummy"
        graph_manager.graphs[GraphType.STANDARD] = "dummy"
        
        assert len(graph_manager.graphs) == 2
        
        # Clear graphs
        graph_manager.clear_graphs()
        
        assert len(graph_manager.graphs) == 0


class TestGraphCapture:
    """Test actual graph capture functionality."""
    
    def test_graph_capture_basic(self, graph_manager, simple_computation):
        """Test basic graph capture."""
        # Capture a graph
        graph_instance = graph_manager.capture_graph(
            graph_type=GraphType.STANDARD,
            capture_func=simple_computation,
        )
        
        # Verify graph instance
        assert isinstance(graph_instance, GraphInstance)
        assert graph_instance.graph_type == GraphType.STANDARD
        assert graph_instance.graph is not None
        assert graph_instance.capture_time > 0
        assert graph_instance.memory_usage > 0
        
        # Verify graph is stored
        assert GraphType.STANDARD in graph_manager.graphs
        assert graph_manager.graphs[GraphType.STANDARD] == graph_instance
    
    def test_graph_execution(self, graph_manager, simple_computation):
        """Test graph execution after capture."""
        # Capture graph first
        graph_manager.capture_graph(
            graph_type=GraphType.SHORT,
            capture_func=simple_computation,
        )
        
        # Create test inputs
        batch_size = 1
        seq_len = 512
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        y = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        
        # Execute graph
        outputs = graph_manager.execute_graph(GraphType.SHORT, [x, y])
        
        # Verify outputs
        assert isinstance(outputs, list)
        assert len(outputs) == 1  # simple_computation returns single tensor
        assert outputs[0].shape == (batch_size, 1, hidden_size)  # After mean(dim=1)
        
        # Verify statistics
        stats = graph_manager.get_stats()
        assert stats["graph_executions"] == 1
        assert stats["graph_hit_rate"] == 1.0  # Only graph executions so far
    
    def test_graph_fallback(self, graph_manager, simple_computation):
        """Test fallback to eager execution."""
        # Don't capture any graphs
        
        # Create test inputs
        x = torch.randn(1, 128, 4096, device='cuda')
        y = torch.randn(1, 128, 4096, device='cuda')
        
        # Execute with fallback (should use eager)
        outputs = graph_manager.execute_with_graph_fallback(
            seq_len=128,
            eager_func=simple_computation,
            inputs=[x, y]
        )
        
        # Verify outputs
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        
        # Verify statistics
        stats = graph_manager.get_stats()
        assert stats["eager_executions"] == 1
        assert stats["graph_hit_rate"] == 0.0  # Only eager executions
    
    def test_multiple_graph_types(self, graph_manager, simple_computation):
        """Test capturing and using multiple graph types."""
        # Capture graphs for different context lengths
        for graph_type in [GraphType.SHORT, GraphType.STANDARD, GraphType.LONG]:
            graph_manager.capture_graph(
                graph_type=graph_type,
                capture_func=simple_computation,
            )
        
        # Verify all graphs captured
        assert len(graph_manager.graphs) == 3
        assert GraphType.SHORT in graph_manager.graphs
        assert GraphType.STANDARD in graph_manager.graphs
        assert GraphType.LONG in graph_manager.graphs
        
        # Test execution with different sequence lengths
        test_cases = [
            (256, GraphType.SHORT),
            (512, GraphType.SHORT),
            (600, GraphType.STANDARD),
            (1024, GraphType.STANDARD),
            (3000, GraphType.LONG),
            (4096, GraphType.LONG),
        ]
        
        for seq_len, expected_type in test_cases:
            graph_type = graph_manager.get_best_fit_graph(seq_len)
            assert graph_type == expected_type


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def test_benchmark_graph_vs_eager(self, graph_manager, simple_computation):
        """Test benchmark function."""
        # Capture graph first
        graph_manager.capture_graph(
            graph_type=GraphType.STANDARD,
            capture_func=simple_computation,
        )
        
        # Run benchmark
        results = benchmark_graph_vs_eager(
            graph_manager=graph_manager,
            eager_func=simple_computation,
            seq_len=1024,
            num_iterations=10  # Small number for quick test
        )
        
        # Verify results structure
        assert "seq_len" in results
        assert results["seq_len"] == 1024
        
        assert "eager_avg_ms" in results
        assert results["eager_avg_ms"] > 0
        
        assert "eager_std_ms" in results
        assert results["eager_std_ms"] >= 0
        
        assert "num_iterations" in results
        assert results["num_iterations"] == 10
        
        # Should have graph results since we captured a graph
        assert "graph_avg_ms" in results
        assert results["graph_avg_ms"] > 0
        
        assert "speedup" in results
        assert results["speedup"] > 0  # Graph should be faster
        
        assert "graph_type" in results
        assert results["graph_type"] == "STANDARD"
    
    def test_benchmark_no_graph(self, graph_manager, simple_computation):
        """Test benchmark without captured graph."""
        # Don't capture any graphs
        
        # Run benchmark
        results = benchmark_graph_vs_eager(
            graph_manager=graph_manager,
            eager_func=simple_computation,
            seq_len=1024,
            num_iterations=5
        )
        
        # Should only have eager results
        assert "seq_len" in results
        assert "eager_avg_ms" in results
        assert "eager_std_ms" in results
        
        # Should not have graph results
        assert "graph_avg_ms" not in results
        assert "speedup" not in results
        assert "graph_type" not in results


class TestContextLengthBuckets:
    """Test context length bucket functionality."""
    
    def test_create_context_length_buckets(self):
        """Test bucket creation."""
        buckets = create_context_length_buckets()
        
        assert len(buckets) == 4
        
        # Check all graph types are present
        assert GraphType.SHORT in buckets
        assert GraphType.STANDARD in buckets
        assert GraphType.MEDIUM in buckets
        assert GraphType.LONG in buckets
        
        # Check values
        assert buckets[GraphType.SHORT] == 512
        assert buckets[GraphType.STANDARD] == 1024
        assert buckets[GraphType.MEDIUM] == 2048
        assert buckets[GraphType.LONG] == 4096


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])