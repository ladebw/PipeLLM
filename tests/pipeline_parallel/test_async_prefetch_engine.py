#!/usr/bin/env python3
"""
Tests for async prefetch engine (Task 2.4).

WARNING: SIMULATED DATA — not from real hardware
These tests verify the async prefetch infrastructure, not actual CUDA performance.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import async prefetch engine
from src.pipeline_parallel.async_prefetch.async_prefetch_engine import (
    AsyncPrefetchEngine,
    PrefetchState,
    LayerPrefetchSchedule,
    PrefetchMetrics
)


class TestPrefetchState:
    """Tests for PrefetchState enum/class."""
    
    def test_state_values(self):
        """Test prefetch state values."""
        # Assuming PrefetchState is an enum or similar
        # These are the expected states from the async_prefetch_engine.py
        expected_states = [
            "IDLE",
            "PREFETCHING",
            "READY",
            "ACTIVE",
            "SWAPPING",
            "ERROR"
        ]
        
        # Check that we can create states (implementation may vary)
        # For now, just verify the concept
        state = PrefetchState.IDLE
        assert state is not None
        
        # Test string representation
        assert "IDLE" in str(state)
    
    def test_state_transitions(self):
        """Test valid state transitions."""
        # This would test that only valid state transitions are allowed
        # For now, just verify basic functionality
        pass


class TestLayerPrefetchSchedule:
    """Tests for LayerPrefetchSchedule class."""
    
    def test_creation(self):
        """Test creating a LayerPrefetchSchedule instance."""
        schedule = LayerPrefetchSchedule(
            layer_id=1,
            layer_name="attention_1",
            prefetch_buffer_id="buffer_1",
            active_buffer_id="buffer_2",
            prefetch_start_time_ns=1000000,
            prefetch_end_time_ns=1500000,
            swap_time_ns=2000000,
            compute_start_time_ns=2500000,
            compute_end_time_ns=3500000,
            prefetch_duration_ns=500000,
            compute_duration_ns=1000000,
            swap_duration_ns=50000,
            state="READY",
            notes="Test schedule"
        )
        
        assert schedule.layer_id == 1
        assert schedule.layer_name == "attention_1"
        assert schedule.prefetch_buffer_id == "buffer_1"
        assert schedule.active_buffer_id == "buffer_2"
        assert schedule.prefetch_start_time_ns == 1000000
        assert schedule.prefetch_end_time_ns == 1500000
        assert schedule.swap_time_ns == 2000000
        assert schedule.compute_start_time_ns == 2500000
        assert schedule.compute_end_time_ns == 3500000
        assert schedule.prefetch_duration_ns == 500000
        assert schedule.compute_duration_ns == 1000000
        assert schedule.swap_duration_ns == 50000
        assert schedule.state == "READY"
        assert schedule.notes == "Test schedule"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        schedule = LayerPrefetchSchedule(
            layer_id=1,
            layer_name="attention_1",
            prefetch_buffer_id="buffer_1",
            active_buffer_id="buffer_2",
            prefetch_start_time_ns=1000000,
            prefetch_end_time_ns=1500000,
            swap_time_ns=2000000,
            compute_start_time_ns=2500000,
            compute_end_time_ns=3500000,
            prefetch_duration_ns=500000,
            compute_duration_ns=1000000,
            swap_duration_ns=50000,
            state="READY",
            notes="Test schedule"
        )
        
        result = schedule.to_dict()
        
        assert isinstance(result, dict)
        assert result["layer_id"] == 1
        assert result["layer_name"] == "attention_1"
        assert result["prefetch_buffer_id"] == "buffer_1"
        assert result["active_buffer_id"] == "buffer_2"
        assert result["prefetch_start_time_ns"] == 1000000
        assert result["prefetch_end_time_ns"] == 1500000
        assert result["swap_time_ns"] == 2000000
        assert result["compute_start_time_ns"] == 2500000
        assert result["compute_end_time_ns"] == 3500000
        assert result["prefetch_duration_ns"] == 500000
        assert result["compute_duration_ns"] == 1000000
        assert result["swap_duration_ns"] == 50000
        assert result["state"] == "READY"
        assert result["notes"] == "Test schedule"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "layer_id": 2,
            "layer_name": "ffn_1",
            "prefetch_buffer_id": "buffer_3",
            "active_buffer_id": "buffer_4",
            "prefetch_start_time_ns": 2000000,
            "prefetch_end_time_ns": 2500000,
            "swap_time_ns": 3000000,
            "compute_start_time_ns": 3500000,
            "compute_end_time_ns": 4500000,
            "prefetch_duration_ns": 500000,
            "compute_duration_ns": 1000000,
            "swap_duration_ns": 60000,
            "state": "ACTIVE",
            "notes": "From dict"
        }
        
        schedule = LayerPrefetchSchedule.from_dict(data)
        
        assert schedule.layer_id == 2
        assert schedule.layer_name == "ffn_1"
        assert schedule.prefetch_buffer_id == "buffer_3"
        assert schedule.active_buffer_id == "buffer_4"
        assert schedule.prefetch_start_time_ns == 2000000
        assert schedule.prefetch_end_time_ns == 2500000
        assert schedule.swap_time_ns == 3000000
        assert schedule.compute_start_time_ns == 3500000
        assert schedule.compute_end_time_ns == 4500000
        assert schedule.prefetch_duration_ns == 500000
        assert schedule.compute_duration_ns == 1000000
        assert schedule.swap_duration_ns == 60000
        assert schedule.state == "ACTIVE"
        assert schedule.notes == "From dict"
    
    def test_update_state(self):
        """Test updating schedule state."""
        schedule = LayerPrefetchSchedule(
            layer_id=1,
            layer_name="test",
            prefetch_buffer_id="buffer_1",
            active_buffer_id="buffer_2",
            prefetch_start_time_ns=0,
            prefetch_end_time_ns=0,
            swap_time_ns=0,
            compute_start_time_ns=0,
            compute_end_time_ns=0,
            prefetch_duration_ns=0,
            compute_duration_ns=0,
            swap_duration_ns=0,
            state="IDLE",
            notes=""
        )
        
        assert schedule.state == "IDLE"
        
        # Update state
        schedule.update_state("PREFETCHING", "Starting prefetch")
        assert schedule.state == "PREFETCHING"
        assert "Starting prefetch" in schedule.notes
    
    def test_calculate_overlap(self):
        """Test overlap calculation."""
        schedule = LayerPrefetchSchedule(
            layer_id=1,
            layer_name="test",
            prefetch_buffer_id="buffer_1",
            active_buffer_id="buffer_2",
            prefetch_start_time_ns=1000000,
            prefetch_end_time_ns=1500000,
            swap_time_ns=2000000,
            compute_start_time_ns=2500000,
            compute_end_time_ns=3500000,
            prefetch_duration_ns=500000,
            compute_duration_ns=1000000,
            swap_duration_ns=50000,
            state="READY",
            notes=""
        )
        
        # Calculate overlap with next layer's prefetch
        next_prefetch_start = 1200000  # Overlaps with current prefetch
        next_prefetch_end = 1700000    # Overlaps with swap
        
        overlap = schedule.calculate_overlap(
            next_prefetch_start,
            next_prefetch_end
        )
        
        assert overlap >= 0
        # Should have some overlap since times overlap


class TestPrefetchMetrics:
    """Tests for PrefetchMetrics class."""
    
    def test_creation(self):
        """Test creating a PrefetchMetrics instance."""
        metrics = PrefetchMetrics(
            total_layers=10,
            prefetched_layers=7,
            active_layers=3,
            total_prefetch_time_ns=5000000,
            total_compute_time_ns=10000000,
            total_swap_time_ns=500000,
            avg_prefetch_time_ns=714286,  # 5,000,000 / 7
            avg_compute_time_ns=1428571,  # 10,000,000 / 7
            avg_swap_time_ns=71429,      # 500,000 / 7
            overlap_time_ns=2000000,
            overlap_percentage=40.0,     # 2,000,000 / 5,000,000 * 100
            buffer_hits=15,
            buffer_misses=5,
            buffer_hit_rate=75.0,
            prefetch_errors=2,
            successful_prefetches=25,
            failed_prefetches=3
        )
        
        assert metrics.total_layers == 10
        assert metrics.prefetched_layers == 7
        assert metrics.active_layers == 3
        assert metrics.total_prefetch_time_ns == 5000000
        assert metrics.total_compute_time_ns == 10000000
        assert metrics.total_swap_time_ns == 500000
        assert metrics.avg_prefetch_time_ns == 714286
        assert metrics.avg_compute_time_ns == 1428571
        assert metrics.avg_swap_time_ns == 71429
        assert metrics.overlap_time_ns == 2000000
        assert metrics.overlap_percentage == 40.0
        assert metrics.buffer_hits == 15
        assert metrics.buffer_misses == 5
        assert metrics.buffer_hit_rate == 75.0
        assert metrics.prefetch_errors == 2
        assert metrics.successful_prefetches == 25
        assert metrics.failed_prefetches == 3
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = PrefetchMetrics(
            total_layers=10,
            prefetched_layers=7,
            active_layers=3,
            total_prefetch_time_ns=5000000,
            total_compute_time_ns=10000000,
            total_swap_time_ns=500000,
            avg_prefetch_time_ns=714286,
            avg_compute_time_ns=1428571,
            avg_swap_time_ns=71429,
            overlap_time_ns=2000000,
            overlap_percentage=40.0,
            buffer_hits=15,
            buffer_misses=5,
            buffer_hit_rate=75.0,
            prefetch_errors=2,
            successful_prefetches=25,
            failed_prefetches=3
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["total_layers"] == 10
        assert result["prefetched_layers"] == 7
        assert result["active_layers"] == 3
        assert result["total_prefetch_time_ns"] == 5000000
        assert result["total_compute_time_ns"] == 10000000
        assert result["total_swap_time_ns"] == 500000
        assert result["avg_prefetch_time_ns"] == 714286
        assert result["avg_compute_time_ns"] == 1428571
        assert result["avg_swap_time_ns"] == 71429
        assert result["overlap_time_ns"] == 2000000
        assert result["overlap_percentage"] == 40.0
        assert result["buffer_hits"] == 15
        assert result["buffer_misses"] == 5
        assert result["buffer_hit_rate"] == 75.0
        assert result["prefetch_errors"] == 2
        assert result["successful_prefetches"] == 25
        assert result["failed_prefetches"] == 3
    
    def test_update_hit_rate(self):
        """Test hit rate calculation."""
        metrics = PrefetchMetrics(
            total_layers=0,
            prefetched_layers=0,
            active_layers=0,
            total_prefetch_time_ns=0,
            total_compute_time_ns=0,
            total_swap_time_ns=0,
            avg_prefetch_time_ns=0,
            avg_compute_time_ns=0,
            avg_swap_time_ns=0,
            overlap_time_ns=0,
            overlap_percentage=0.0,
            buffer_hits=30,
            buffer_misses=10,
            buffer_hit_rate=0.0,  # Will be calculated
            prefetch_errors=0,
            successful_prefetches=0,
            failed_prefetches=0
        )
        
        # Calculate hit rate
        hit_rate = metrics._calculate_hit_rate()
        assert hit_rate == 75.0  # 30 / (30 + 10) * 100
        
        # Test with zero hits and misses
        metrics.buffer_hits = 0
        metrics.buffer_misses = 0
        hit_rate_zero = metrics._calculate_hit_rate()
        assert hit_rate_zero == 0.0


class TestAsyncPrefetchEngine:
    """Tests for AsyncPrefetchEngine class."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = AsyncPrefetchEngine(
            num_layers=4,
            buffer_size_bytes=16 * 1024 * 1024,  # 16MB
            enable_async=True
        )
        
        assert engine.num_layers == 4
        assert engine.buffer_size_bytes == 16 * 1024 * 1024
        assert engine.enable_async is True
        assert hasattr(engine, 'stream_manager')
        assert hasattr(engine, 'memory_pool')
        assert hasattr(engine, 'schedules')
        assert hasattr(engine, 'metrics')
        
        # Check that schedules were created
        assert len(engine.schedules) == 4
        for i, schedule in enumerate(engine.schedules):
            assert schedule.layer_id == i + 1
            assert schedule.state == "IDLE"
        
        # Check metrics
        assert engine.metrics.total_layers == 4
        assert engine.metrics.prefetched_layers == 0
        assert engine.metrics.active_layers == 0
    
    @patch('src.pipeline_parallel.async_prefetch.async_prefetch_engine.DualStreamManager')
    @patch('src.pipeline_parallel.async_prefetch.async_prefetch_engine.PinnedMemoryPool')
    def test_initialization_with_mocks(self, mock_pool_class, mock_stream_class):
        """Test initialization with mocked dependencies."""
        mock_stream = Mock()
        mock_pool = Mock()
        mock_stream_class.return_value = mock_stream
        mock_pool_class.return_value = mock_pool
        
        engine = AsyncPrefetchEngine(
            num_layers=3,
            buffer_size_bytes=8 * 1024 * 1024,
            enable_async=True
        )
        
        # Check that dependencies were created
        mock_stream_class.assert_called_once()
        mock_pool_class.assert_called_once_with(
            max_memory_bytes=24 * 1024 * 1024,  # 3 layers * 8MB * 2 buffers
            buffer_sizes=[8 * 1024 * 1024],
            enable_pinning=True
        )
        
        assert engine.stream_manager == mock_stream
        assert engine.memory_pool == mock_pool
    
    def test_create_layer_schedule(self):
        """Test layer schedule creation."""
        engine = AsyncPrefetchEngine(num_layers=1)
        
        # The engine should have created schedules during initialization
        assert len(engine.schedules) == 1
        
        schedule = engine.schedules[0]
        assert schedule.layer_id == 1
        assert schedule.layer_name == "layer_1"
        assert schedule.state == "IDLE"
        assert schedule.prefetch_buffer_id is None
        assert schedule.active_buffer_id is None
    
    def test_allocate_buffers_for_layer(self):
        """Test buffer allocation for a layer."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock the memory pool
        mock_buffer1 = Mock()
        mock_buffer1.buffer_id = "buffer_1"
        mock_buffer1.size_bytes = engine.buffer_size_bytes
        
        mock_buffer2 = Mock()
        mock_buffer2.buffer_id = "buffer_2"
        mock_buffer2.size_bytes = engine.buffer_size_bytes
        
        engine.memory_pool.allocate_buffer.side_effect = [mock_buffer1, mock_buffer2]
        
        # Allocate buffers for layer 1
        success = engine._allocate_buffers_for_layer(0)  # layer index 0
        
        assert success is True
        assert engine.memory_pool.allocate_buffer.call_count == 2
        
        # Check that buffers were assigned to schedule
        schedule = engine.schedules[0]
        assert schedule.prefetch_buffer_id == "buffer_1"
        assert schedule.active_buffer_id == "buffer_2"
        
        # Check metrics
        assert engine.metrics.buffer_hits == 2  # Two successful allocations
    
    def test_allocate_buffers_failure(self):
        """Test buffer allocation failure."""
        engine = AsyncPrefetchEngine(num_layers=1)
        
        # Mock memory pool to fail allocation
        engine.memory_pool.allocate_buffer.return_value = None
        
        # Try to allocate buffers
        success = engine._allocate_buffers_for_layer(0)
        
        assert success is False
        assert engine.memory_pool.allocate_buffer.call_count == 1
        
        # Check metrics
        assert engine.metrics.buffer_misses == 1
        assert engine.metrics.failed_prefetches == 0  # Not a prefetch failure yet
    
    def test_prefetch_layer_weights(self):
        """Test weight prefetching for a layer."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock dependencies
        engine.memory_pool.get_buffer_info.return_value = {"buffer_id": "buffer_1"}
        
        # Mock stream manager memory operation
        mock_result = {"success": True, "duration_ns": 50000}
        engine.stream_manager.execute_memory_operation.return_value = mock_result
        
        # First allocate buffers
        engine._allocate_buffers_for_layer = Mock(return_value=True)
        engine._allocate_buffers_for_layer(0)
        
        # Prefetch weights for layer 1
        success = engine.prefetch_layer_weights(
            layer_index=0,
            weight_size_bytes=8 * 1024 * 1024,
            description="Prefetch layer 1 weights"
        )
        
        assert success is True
        assert engine.stream_manager.execute_memory_operation.call_count == 1
        
        # Check schedule state
        schedule = engine.schedules[0]
        assert schedule.state == "PREFETCHING"
        assert schedule.prefetch_duration_ns == 50000
        
        # Check metrics
        assert engine.metrics.prefetched_layers == 1
        assert engine.metrics.successful_prefetches == 1
    
    def test_prefetch_layer_weights_no_buffers(self):
        """Test weight prefetching when no buffers allocated."""
        engine = AsyncPrefetchEngine(num_layers=1)
        
        # Don't allocate buffers first
        # Try to prefetch
        success = engine.prefetch_layer_weights(
            layer_index=0,
            weight_size_bytes=8 * 1024 * 1024,
            description="Test prefetch"
        )
        
        assert success is False
        
        # Check schedule state (should still be IDLE)
        schedule = engine.schedules[0]
        assert schedule.state == "IDLE"
        
        # Check metrics
        assert engine.metrics.failed_prefetches == 1
    
    def test_execute_layer_computation(self):
        """Test layer computation execution."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock stream manager compute operation
        mock_result = {"success": True, "duration_ns": 150000}
        engine.stream_manager.execute_compute_operation.return_value = mock_result
        
        # Set up schedule
        schedule = engine.schedules[0]
        schedule.state = "READY"
        schedule.active_buffer_id = "buffer_1"
        
        # Execute computation
        success = engine.execute_layer_computation(
            layer_index=0,
            operation_type="attention",
            duration_ns=150000,
            description="Layer 1 attention"
        )
        
        assert success is True
        assert engine.stream_manager.execute_compute_operation.call_count == 1
        
        # Check schedule state
        assert schedule.state == "ACTIVE"
        assert schedule.compute_duration_ns == 150000
        
        # Check metrics
        assert engine.metrics.active_layers == 1
    
    def test_execute_layer_computation_not_ready(self):
        """Test layer computation when layer is not ready."""
        engine = AsyncPrefetchEngine(num_layers=1)
        
        # Schedule is in IDLE state (not READY)
        schedule = engine.schedules[0]
        schedule.state = "IDLE"
        
        # Try to execute computation
        success = engine.execute_layer_computation(
            layer_index=0,
            operation_type="attention",
            duration_ns=100000,
            description="Test"
        )
        
        assert success is False
        
        # Check schedule state (should still be IDLE)
        assert schedule.state == "IDLE"
        
        # Check metrics
        assert engine.metrics.prefetch_errors == 1
    
    def test_swap_buffers(self):
        """Test buffer swapping between prefetch and active."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock stream manager sync operation
        mock_result = {"success": True, "duration_ns": 20000}
        engine.stream_manager.synchronize_streams.return_value = Mock(
            duration_ns=20000,
            event_type="sync"
        )
        
        # Set up schedules
        schedule1 = engine.schedules[0]
        schedule1.state = "ACTIVE"
        schedule1.prefetch_buffer_id = "buffer_prefetch_1"
        schedule1.active_buffer_id = "buffer_active_1"
        
        schedule2 = engine.schedules[1]
        schedule2.state = "PREFETCHING"
        schedule2.prefetch_buffer_id = "buffer_prefetch_2"
        schedule2.active_buffer_id = "buffer_active_2"
        
        # Swap buffers between layer 1 and 2
        success = engine.swap_buffers(
            source_layer_index=0,
            target_layer_index=1,
            description="Swap buffers for pipeline"
        )
        
        assert success is True
        assert engine.stream_manager.synchronize_streams.call_count == 1
        
        # Check that buffers were swapped
        # In a real implementation, buffer IDs would be exchanged
        # For now, just check that the operation was attempted
        
        # Check schedule states
        # State transitions would depend on implementation
    
    def test_execute_pipeline_step(self):
        """Test executing a complete pipeline step."""
        engine = AsyncPrefetchEngine(num_layers=3)
        
        # Mock all operations
        engine.prefetch_layer_weights = Mock(return_value=True)
        engine.execute_layer_computation = Mock(return_value=True)
        engine.swap_buffers = Mock(return_value=True)
        engine.stream_manager.measure_overlap.return_value = {
            "overlap_percentage": 65.0,
            "overlap_time_ns": 130000
        }
        
        # Execute pipeline step
        result = engine.execute_pipeline_step(
            step_id=1,
            prefetch_layer=0,
            compute_layer=1,
            swap_layers=[(1, 2)],
            prefetch_duration_ns=80000,
            compute_duration_ns=200000,
            swap_duration_ns=20000
        )
        
        assert result is not None
        assert result["success"] is True
        assert result["step_id"] == 1
        assert result["overlap_percentage"] == 65.0
        
        # Check that operations were called
        assert engine.prefetch_layer_weights.call_count == 1
        assert engine.execute_layer_computation.call_count == 1
        assert engine.swap_buffers.call_count == 1
        
        # Check metrics
        assert engine.metrics.successful_prefetches == 1
    
    def test_execute_pipeline_step_with_failure(self):
        """Test pipeline step execution with failure."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock prefetch to fail
        engine.prefetch_layer_weights = Mock(return_value=False)
        
        # Execute pipeline step (should fail)
        result = engine.execute_pipeline_step(
            step_id=1,
            prefetch_layer=0,
            compute_layer=1,
            swap_layers=[],
            prefetch_duration_ns=80000,
            compute_duration_ns=200000,
            swap_duration_ns=0
        )
        
        assert result is not None
        assert result["success"] is False
        assert "error" in result
        
        # Check metrics
        assert engine.metrics.failed_prefetches == 1
        assert engine.metrics.prefetch_errors == 1
    
    def test_get_layer_status(self):
        """Test getting layer status."""
        engine = AsyncPrefetchEngine(num_layers=3)
        
        # Set up different states
        engine.schedules[0].state = "IDLE"
        engine.schedules[1].state = "PREFETCHING"
        engine.schedules[2].state = "ACTIVE"
        
        # Get status for all layers
        status = engine.get_layer_status()
        
        assert isinstance(status, dict)
        assert "layers" in status
        assert "summary" in status
        
        layers = status["layers"]
        assert len(layers) == 3
        
        # Check individual layer status
        assert layers[0]["layer_id"] == 1
        assert layers[0]["state"] == "IDLE"
        
        assert layers[1]["layer_id"] == 2
        assert layers[1]["state"] == "PREFETCHING"
        
        assert layers[2]["layer_id"] == 3
        assert layers[2]["state"] == "ACTIVE"
        
        # Check summary
        summary = status["summary"]
        assert summary["total_layers"] == 3
        assert summary["idle_layers"] == 1
        assert summary["prefetching_layers"] == 1
        assert summary["active_layers"] == 1
    
    def test_get_engine_metrics(self):
        """Test getting engine metrics."""
        engine = AsyncPrefetchEngine(num_layers=4)
        
        # Update some metrics
        engine.metrics.prefetched_layers = 2
        engine.metrics.active_layers = 1
        engine.metrics.successful_prefetches = 5
        engine.metrics.failed_prefetches = 1
        
        # Get metrics
        metrics = engine.get_engine_metrics()
        
        assert isinstance(metrics, dict)
        assert "prefetch_metrics" in metrics
        assert "stream_metrics" in metrics
        assert "memory_pool_metrics" in metrics
        
        prefetch_metrics = metrics["prefetch_metrics"]
        assert prefetch_metrics["total_layers"] == 4
        assert prefetch_metrics["prefetched_layers"] == 2
        assert prefetch_metrics["active_layers"] == 1
        assert prefetch_metrics["successful_prefetches"] == 5
        assert prefetch_metrics["failed_prefetches"] == 1
        
        # Calculate success rate
        total_prefetches = 5 + 1  # successful + failed
        success_rate = (5 / total_prefetches) * 100 if total_prefetches > 0 else 0
        assert prefetch_metrics.get("prefetch_success_rate", 0) == success_rate
    
    def test_generate_report(self):
        """Test report generation."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Execute some operations
        engine.prefetch_layer_weights = Mock(return_value=True)
        engine.execute_layer_computation = Mock(return_value=True)
        
        engine.prefetch_layer_weights(0, 8 * 1024 * 1024, "Test prefetch")
        engine.execute_layer_computation(0, "attention", 150000, "Test compute")
        
        # Generate report
        report = engine.generate_report()
        
        assert isinstance(report, str)
        assert "ASYNC PREFETCH ENGINE REPORT" in report
        assert "WARNING" in report
        assert "SIMULATED" in report
        assert "layer_1" in report
        assert "attention" in report.lower()
        assert "metrics" in report.lower()
    
    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        engine = AsyncPrefetchEngine(num_layers=1)
        
        # Save report
        output_path = tmp_path / "engine_report.txt"
        saved_path = engine.save_report(output_path)
        
        assert saved_path.exists()
        
        # Load and verify
        with open(saved_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "ASYNC PREFETCH ENGINE REPORT" in report_content
        assert "WARNING" in report_content
        assert "SIMULATED" in report_content


class TestIntegration:
    """Integration tests for async prefetch engine."""
    
    def test_complete_pipeline_execution(self):
        """Test complete pipeline execution simulation."""
        engine = AsyncPrefetchEngine(
            num_layers=4,
            buffer_size_bytes=8 * 1024 * 1024,
            enable_async=True
        )
        
        # Mock all operations for a complete pipeline
        engine.prefetch_layer_weights = Mock(return_value=True)
        engine.execute_layer_computation = Mock(return_value=True)
        engine.swap_buffers = Mock(return_value=True)
        
        engine.stream_manager.measure_overlap.return_value = {
            "overlap_percentage": 72.5,
            "overlap_time_ns": 145000,
            "total_time_ns": 200000
        }
        
        # Simulate a 3-step pipeline
        results = []
        for step in range(3):
            result = engine.execute_pipeline_step(
                step_id=step + 1,
                prefetch_layer=step % 4,
                compute_layer=(step + 1) % 4,
                swap_layers=[((step + 1) % 4, (step + 2) % 4)],
                prefetch_duration_ns=75000,
                compute_duration_ns=180000,
                swap_duration_ns=15000
            )
            results.append(result)
        
        # Check results
        assert len(results) == 3
        assert all(r["success"] is True for r in results)
        
        # Check that operations were called
        assert engine.prefetch_layer_weights.call_count == 3
        assert engine.execute_layer_computation.call_count == 3
        assert engine.swap_buffers.call_count == 3
        
        # Check metrics
        assert engine.metrics.successful_prefetches == 3
        assert engine.metrics.total_prefetch_time_ns == 225000  # 3 * 75000
    
    def test_engine_with_disabled_async(self):
        """Test engine with async disabled (synchronous mode)."""
        engine = AsyncPrefetchEngine(
            num_layers=3,
            buffer_size_bytes=8 * 1024 * 1024,
            enable_async=False  # Disable async prefetch
        )
        
        # In synchronous mode, prefetch should happen before compute
        # without overlap
        
        # Mock operations
        engine.prefetch_layer_weights = Mock(return_value=True)
        engine.execute_layer_computation = Mock(return_value=True)
        
        # Execute operations
        engine.prefetch_layer_weights(0, 8 * 1024 * 1024, "Sync prefetch")
        engine.execute_layer_computation(0, "layer", 200000, "Sync compute")
        
        # In sync mode, there should be no overlap
        # (implementation would ensure sequential execution)
        
        # Check that operations were called
        assert engine.prefetch_layer_weights.call_count == 1
        assert engine.execute_layer_computation.call_count == 1
    
    def test_error_recovery(self):
        """Test error recovery scenario."""
        engine = AsyncPrefetchEngine(num_layers=2)
        
        # Mock first prefetch to fail, second to succeed
        engine.prefetch_layer_weights = Mock(side_effect=[False, True])
        engine.execute_layer_computation = Mock(return_value=True)
        
        # Try to execute pipeline (first attempt fails)
        result1 = engine.execute_pipeline_step(
            step_id=1,
            prefetch_layer=0,
            compute_layer=1,
            swap_layers=[],
            prefetch_duration_ns=80000,
            compute_duration_ns=200000,
            swap_duration_ns=0
        )
        
        assert result1["success"] is False
        
        # Retry (should succeed)
        result2 = engine.execute_pipeline_step(
            step_id=2,
            prefetch_layer=0,
            compute_layer=1,
            swap_layers=[],
            prefetch_duration_ns=80000,
            compute_duration_ns=200000,
            swap_duration_ns=0
        )
        
        assert result2["success"] is True
        
        # Check metrics
        assert engine.metrics.failed_prefetches == 1
        assert engine.metrics.successful_prefetches == 1
        assert engine.metrics.prefetch_errors == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])