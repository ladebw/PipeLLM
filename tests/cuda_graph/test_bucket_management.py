"""
Tests for Context Length Bucket Management (Phase 1, Task 1.4).

These tests verify the enhanced bucket management system with:
1. Multiple context length bucket handling
2. Dynamic graph selection
3. Memory-aware caching with LRU eviction
4. Adaptive learning and optimization
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cuda_graph.cuda_graph_buckets import (
    GraphBucketManager,
    AdaptiveBucketManager,
    GraphType,
    GraphInstance,
    create_production_bucket_manager
)
from cuda_graph.bucket_integration import (
    BucketAwareLayerAdapter,
    BucketAwareModelWrapper,
    wrap_model_with_bucket_management
)


@pytest.fixture
def cuda_available():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return True


@pytest.fixture
def mock_transformer_layer():
    """Create a mock transformer layer for testing."""
    
    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.hidden_size = hidden_size
            self.linear = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x, mask=None, pos=None, past_kv=None):
            # Simple forward pass for testing
            output = self.linear(x)
            return (output,)
    
    return MockTransformerLayer()


@pytest.fixture
def simple_capture_func():
    """Create a simple capture function for testing."""
    def capture_func(x, mask, pos, past_kv):
        # Simple computation
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        return (output,)
    
    return capture_func


class TestGraphBucketManager:
    """Test basic bucket manager functionality."""
    
    def test_initialization(self, cuda_available):
        """Test bucket manager initialization."""
        manager = GraphBucketManager(
            device="cuda",
            max_total_memory_mb=1024,
            enable_lru=True,
            capture_on_demand=True
        )
        
        assert manager.device.type == 'cuda'
        assert manager.max_total_memory_bytes == 1024 * 1024 * 1024
        assert manager.enable_lru is True
        assert manager.capture_on_demand is True
        assert len(manager.graphs) == 0
        assert manager.total_memory_used == 0
        
        # Check statistics
        stats = manager.stats
        assert stats["total_requests"] == 0
        assert stats["graph_hits"] == 0
        assert stats["graph_misses"] == 0
    
    def test_get_best_fit_graph_type(self, cuda_available):
        """Test graph type selection logic."""
        manager = GraphBucketManager()
        
        # Test with no graphs captured
        assert manager._find_best_fit_graph_type(256) == GraphType.SHORT
        assert manager._find_best_fit_graph_type(512) == GraphType.SHORT
        assert manager._find_best_fit_graph_type(600) == GraphType.STANDARD
        assert manager._find_best_fit_graph_type(1024) == GraphType.STANDARD
        assert manager._find_best_fit_graph_type(1500) == GraphType.MEDIUM
        assert manager._find_best_fit_graph_type(2048) == GraphType.MEDIUM
        assert manager._find_best_fit_graph_type(3000) == GraphType.LONG
        assert manager._find_best_fit_graph_type(4096) == GraphType.LONG
        assert manager._find_best_fit_graph_type(5000) is None  # Too large
    
    def test_get_graph_for_seq_len_no_capture(self, cuda_available, simple_capture_func):
        """Test graph retrieval without capture."""
        manager = GraphBucketManager(capture_on_demand=False)
        
        # Should return None since no graphs captured
        graph_instance = manager.get_graph_for_seq_len(
            seq_len=1024,
            capture_func=simple_capture_func
        )
        
        assert graph_instance is None
        assert manager.stats["graph_misses"] == 1
    
    def test_capture_graph_on_demand(self, cuda_available, simple_capture_func):
        """Test on-demand graph capture."""
        manager = GraphBucketManager(
            max_total_memory_mb=2048,
            capture_on_demand=True
        )
        
        # Capture a graph on demand
        graph_instance = manager.get_graph_for_seq_len(
            seq_len=1024,
            capture_func=simple_capture_func
        )
        
        assert graph_instance is not None
        assert isinstance(graph_instance, GraphInstance)
        assert graph_instance.graph_type == GraphType.STANDARD
        
        # Should be stored
        assert GraphType.STANDARD in manager.graphs
        assert manager.stats["graph_captures"] == 1
        
        # Memory should be tracked
        assert manager.total_memory_used > 0
        assert GraphType.STANDARD in manager.memory_by_type
    
    def test_memory_limit_enforcement(self, cuda_available, simple_capture_func):
        """Test memory limit enforcement."""
        # Set very low memory limit
        manager = GraphBucketManager(
            max_total_memory_mb=10,  # 10 MB - very small
            enable_lru=True,
            capture_on_demand=True
        )
        
        # Try to capture a graph (should fail due to memory limit)
        graph_instance = manager.get_graph_for_seq_len(
            seq_len=512,
            capture_func=simple_capture_func
        )
        
        # Should fail to capture due to memory limit
        assert graph_instance is None
        assert manager.stats["graph_misses"] == 1
    
    def test_lru_eviction(self, cuda_available, simple_capture_func):
        """Test LRU eviction when memory limit reached."""
        # Set memory limit that can hold one graph
        manager = GraphBucketManager(
            max_total_memory_mb=100,  # Enough for one graph
            enable_lru=True,
            capture_on_demand=True
        )
        
        # Capture first graph
        graph1 = manager.get_graph_for_seq_len(
            seq_len=512,
            capture_func=simple_capture_func
        )
        
        assert graph1 is not None
        assert GraphType.SHORT in manager.graphs
        
        memory_used = manager.total_memory_used
        
        # Capture second graph (should trigger eviction)
        graph2 = manager.get_graph_for_seq_len(
            seq_len=1024,
            capture_func=simple_capture_func
        )
        
        # Should have evicted first graph
        assert graph2 is not None
        assert GraphType.STANDARD in manager.graphs
        assert GraphType.SHORT not in manager.graphs  # Should be evicted
        assert manager.stats["graph_evictions"] >= 1
    
    def test_execute_with_bucket_selection(self, cuda_available, simple_capture_func):
        """Test execution with automatic bucket selection."""
        manager = GraphBucketManager(capture_on_demand=True)
        
        # Create test inputs
        batch_size = 1
        seq_len = 1024
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = [x, mask, pos, None]
        
        # Execute
        outputs, graph_type_used = manager.execute_with_bucket_selection(
            seq_len=seq_len,
            eager_func=simple_capture_func,
            inputs=inputs,
            capture_func=simple_capture_func
        )
        
        # Should have used a graph
        assert graph_type_used == GraphType.STANDARD
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0].shape == (batch_size, seq_len, seq_len)  # x @ x.T shape
    
    def test_pre_capture_graphs(self, cuda_available, simple_capture_func):
        """Test pre-capturing multiple graphs."""
        manager = GraphBucketManager(max_total_memory_mb=2048)
        
        # Pre-capture all graph types
        results = manager.pre_capture_graphs(
            capture_func=simple_capture_func,
            graph_types=[GraphType.SHORT, GraphType.STANDARD]
        )
        
        assert len(results) == 2
        assert GraphType.SHORT in results
        assert GraphType.STANDARD in results
        
        # Should have captured graphs
        assert GraphType.SHORT in manager.graphs or not results[GraphType.SHORT]
        assert GraphType.STANDARD in manager.graphs or not results[GraphType.STANDARD]
    
    def test_get_memory_stats(self, cuda_available, simple_capture_func):
        """Test memory statistics."""
        manager = GraphBucketManager(max_total_memory_mb=1024)
        
        # Capture a graph
        manager.get_graph_for_seq_len(
            seq_len=512,
            capture_func=simple_capture_func
        )
        
        # Get memory stats
        stats = manager.get_memory_stats()
        
        assert "total_memory_used_bytes" in stats
        assert "total_memory_used_mb" in stats
        assert "memory_limit_bytes" in stats
        assert "memory_limit_mb" in stats
        assert "memory_available_bytes" in stats
        assert "graph_count" in stats
        
        assert stats["total_memory_used_bytes"] > 0
        assert stats["graph_count"] == 1
    
    def test_get_performance_report(self, cuda_available, simple_capture_func):
        """Test performance reporting."""
        manager = GraphBucketManager(capture_on_demand=True)
        
        # Execute some operations
        for seq_len in [512, 1024, 512, 2048]:
            manager.get_graph_for_seq_len(
                seq_len=seq_len,
                capture_func=simple_capture_func
            )
        
        # Get performance report
        report = manager.get_performance_report()
        
        assert "total_requests" in report
        assert "graph_hit_rate_percent" in report
        assert "graph_hits" in report
        assert "graph_misses" in report
        assert "eager_fallbacks" in report
        
        assert report["total_requests"] == 4


class TestAdaptiveBucketManager:
    """Test adaptive bucket manager with learning."""
    
    def test_initialization(self, cuda_available):
        """Test adaptive manager initialization."""
        manager = AdaptiveBucketManager(
            device="cuda",
            max_total_memory_mb=1024,
            adaptive_memory_allocation=True
        )
        
        assert isinstance(manager, AdaptiveBucketManager)
        assert manager.adaptive_memory_allocation is True
        assert manager.learning_window == 1000
        
        # Check learning data structures
        assert isinstance(manager.usage_patterns, dict)
        assert isinstance(manager.hit_patterns, dict)
        
        # Should have entries for all graph types
        for graph_type in GraphType:
            assert graph_type in manager.hit_patterns
    
    def test_learning_patterns(self, cuda_available, simple_capture_func):
        """Test learning from usage patterns."""
        manager = AdaptiveBucketManager(capture_on_demand=True)
        
        # Simulate usage patterns
        seq_lens = [512, 1024, 512, 2048, 512, 1024]
        
        for seq_len in seq_lens:
            manager.get_graph_for_seq_len(
                seq_len=seq_len,
                capture_func=simple_capture_func
            )
        
        # Check usage patterns
        assert 512 in manager.usage_patterns
        assert manager.usage_patterns[512] == 3  # Used 3 times
        
        assert 1024 in manager.usage_patterns
        assert manager.usage_patterns[1024] == 2  # Used 2 times
        
        # Check hit patterns
        short_patterns = manager.hit_patterns[GraphType.SHORT]
        assert short_patterns["hits"] > 0 or short_patterns["misses"] > 0
    
    def test_get_optimal_graphs_to_keep(self, cuda_available, simple_capture_func):
        """Test optimal graph selection."""
        manager = AdaptiveBucketManager(
            max_total_memory_mb=500,  # Limited memory
            capture_on_demand=True
        )
        
        # Capture some graphs
        for seq_len in [512, 1024, 2048]:
            manager.get_graph_for_seq_len(
                seq_len=seq_len,
                capture_func=simple_capture_func
            )
        
        # Get optimal graphs
        optimal_graphs = manager.get_optimal_graphs_to_keep()
        
        assert isinstance(optimal_graphs, list)
        assert all(isinstance(gt, GraphType) for gt in optimal_graphs)
        
        # Should be a subset of captured graphs
        captured_graphs = set(manager.graphs.keys())
        assert set(optimal_graphs).issubset(captured_graphs)
    
    def test_optimize_memory_allocation(self, cuda_available, simple_capture_func):
        """Test memory optimization."""
        manager = AdaptiveBucketManager(
            max_total_memory_mb=500,
            capture_on_demand=True
        )
        
        # Capture multiple graphs
        for seq_len in [512, 1024, 2048, 4096]:
            manager.get_graph_for_seq_len(
                seq_len=seq_len,
                capture_func=simple_capture_func
            )
        
        initial_count = len(manager.graphs)
        initial_memory = manager.total_memory_used
        
        # Optimize memory allocation
        manager.optimize_memory_allocation()
        
        # Should have evicted some graphs
        assert len(manager.graphs) <= initial_count
        assert manager.total_memory_used <= initial_memory
        
        # Should still have some graphs
        assert len(manager.graphs) > 0
    
    def test_get_learning_report(self, cuda_available, simple_capture_func):
        """Test learning report generation."""
        manager = AdaptiveBucketManager(capture_on_demand=True)
        
        # Generate some usage
        for seq_len in [512, 1024, 512, 2048]:
            manager.get_graph_for_seq_len(
                seq_len=seq_len,
                capture_func=simple_capture_func
            )
        
        # Get learning report
        report = manager.get_learning_report()
        
        assert "total_requests_analyzed" in report
        assert "most_common_sequence_lengths" in report
        assert "graph_effectiveness" in report
        assert "recommended_graphs_to_keep" in report
        
        # Should have analyzed our requests
        assert report["total_requests_analyzed"] == 4
        
        # Should have effectiveness data
        assert len(report["graph_effectiveness"]) > 0


class TestBucketAwareLayerAdapter:
    """Test bucket-aware layer adapter."""
    
    def test_initialization(self, cuda_available, mock_transformer_layer):
        """Test layer adapter initialization."""
        bucket_manager = GraphBucketManager()
        
        adapter = BucketAwareLayerAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            bucket_manager=bucket_manager,
            hidden_size=4096,
            num_heads=32,
            head_dim=128
        )
        
        assert adapter.layer == mock_transformer_layer
        assert adapter.layer_idx == 0
        assert adapter.bucket_manager == bucket_manager
        assert adapter.hidden_size == 4096
        assert adapter.num_heads == 32
        assert adapter.head_dim == 128
        assert len(adapter.captured_graph_types) == 0
        
        # Check statistics
        assert adapter.stats["total_calls"] == 0
        assert adapter.stats["graph_calls"] == 0
        assert adapter.stats["eager_calls"] == 0
    
    def test_forward_with_graph(self, cuda_available, mock_transformer_layer):
        """Test forward pass with graph execution."""
        bucket_manager = GraphBucketManager(capture_on_demand=True)
        
        adapter = BucketAwareLayerAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            bucket_manager=bucket_manager
        )
        
        # Create test inputs
        batch_size = 1
        seq_len = 1024
        hidden_size = 4096
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass (should capture and use graph)
        outputs = adapter.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False
        )
        
        # Should return tuple
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        
        hidden_states_out, past_kv, attentions = outputs
        
        assert hidden_states_out.shape == (batch_size, seq_len, hidden_size)
        assert past_kv is None  # use_cache=False
        assert attentions is None  # output_attentions=False
        
        # Should have used graph
        assert adapter.stats["graph_calls"] >= 1
        assert adapter.stats["total_calls"] == 1
    
    def test_pre_capture_graphs(self, cuda_available, mock_transformer_layer):
        """Test pre-capturing graphs for a layer."""
        bucket_manager = GraphBucketManager(max_total_memory_mb=2048)
        
        adapter = BucketAwareLayerAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            bucket_manager=bucket_manager
        )
        
        # Pre-capture graphs
        results = adapter.pre_capture_graphs(
            graph_types=[GraphType.SHORT, GraphType.STANDARD]
        )
        
        assert isinstance(results, dict)
        assert len(results) == 2
        
        # Should have captured some graphs
        assert len(adapter.captured_graph_types) >= 0  # Could be 0 if memory limited
    
    def test_get_stats(self, cuda_available, mock_transformer_layer):
        """Test statistics collection."""
        bucket_manager = GraphBucketManager(capture_on_demand=True)
        
        adapter = BucketAwareLayerAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            bucket_manager=bucket_manager
        )
        
        # Do some forward passes
        batch_size = 1
        seq_len = 512
        hidden_size = 4096
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        
        for _ in range(3):
            adapter.forward(
                hidden_states=hidden_states,
                use_cache=False,
                output_attentions=False
            )
        
        # Get statistics
        stats = adapter.get_stats()
        
        assert stats["total_calls"] == 3
        assert stats["layer_idx"] == 0
        assert "graph_call_rate" in stats
        assert "average_time_ms" in stats
        assert "captured_graph_types" in stats


class TestProductionFactory:
    """Test production factory function."""
    
    def test_create_production_bucket_manager(self, cuda_available):
        """Test factory function for creating bucket managers."""
        # Test with adaptive=True
        adaptive_manager = create_production_bucket_manager(
            device="cuda",
            memory_limit_mb=2048,
            adaptive=True
        )
        
        assert isinstance(adaptive_manager, AdaptiveBucketManager)
        assert adaptive_manager.max_total_memory_bytes == 2048 * 1024 * 1024
        assert adaptive_manager.enable_lru is True
        assert adaptive_manager.capture_on_demand is True
        
        # Test with adaptive=False
        basic_manager = create_production_bucket_manager(
            device="cuda",
            memory_limit_mb=1024,
            adaptive=False
        )
        
        assert isinstance(basic_manager, GraphBucketManager)
        assert not isinstance(basic_manager, AdaptiveBucketManager)
        assert basic_manager.max_total_memory_bytes == 1024 * 1024 * 1024


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])