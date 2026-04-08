#!/usr/bin/env python3
"""
Tests for dual stream manager (Task 2.2).

WARNING: SIMULATED DATA — not from real hardware
These tests verify the stream management infrastructure, not actual CUDA performance.
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

# Import dual stream manager
from src.pipeline_parallel.async_prefetch.dual_stream_manager import (
    DualStreamManager,
    StreamEvent,
    StreamMetrics
)


class TestStreamEvent:
    """Tests for StreamEvent class."""
    
    def test_creation(self):
        """Test creating a StreamEvent instance."""
        event = StreamEvent(
            event_id="test_event_1",
            event_type="sync",
            stream_type="compute",
            timestamp_ns=1000000,
            duration_ns=50000,
            description="Test synchronization event"
        )
        
        assert event.event_id == "test_event_1"
        assert event.event_type == "sync"
        assert event.stream_type == "compute"
        assert event.timestamp_ns == 1000000
        assert event.duration_ns == 50000
        assert event.description == "Test synchronization event"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        event = StreamEvent(
            event_id="test_event_1",
            event_type="sync",
            stream_type="compute",
            timestamp_ns=1000000,
            duration_ns=50000,
            description="Test event"
        )
        
        result = event.to_dict()
        
        assert isinstance(result, dict)
        assert result["event_id"] == "test_event_1"
        assert result["event_type"] == "sync"
        assert result["stream_type"] == "compute"
        assert result["timestamp_ns"] == 1000000
        assert result["duration_ns"] == 50000
        assert result["description"] == "Test event"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "event_id": "test_event_2",
            "event_type": "memory",
            "stream_type": "copy",
            "timestamp_ns": 2000000,
            "duration_ns": 75000,
            "description": "Memory transfer event"
        }
        
        event = StreamEvent.from_dict(data)
        
        assert event.event_id == "test_event_2"
        assert event.event_type == "memory"
        assert event.stream_type == "copy"
        assert event.timestamp_ns == 2000000
        assert event.duration_ns == 75000
        assert event.description == "Memory transfer event"


class TestStreamMetrics:
    """Tests for StreamMetrics class."""
    
    def test_creation(self):
        """Test creating a StreamMetrics instance."""
        metrics = StreamMetrics(
            stream_type="compute",
            total_events=10,
            total_time_ns=1000000,
            avg_event_time_ns=100000,
            max_event_time_ns=200000,
            min_event_time_ns=50000,
            sync_events=3,
            memory_events=5,
            compute_events=2,
            idle_time_ns=200000,
            utilization_pct=80.0
        )
        
        assert metrics.stream_type == "compute"
        assert metrics.total_events == 10
        assert metrics.total_time_ns == 1000000
        assert metrics.avg_event_time_ns == 100000
        assert metrics.max_event_time_ns == 200000
        assert metrics.min_event_time_ns == 50000
        assert metrics.sync_events == 3
        assert metrics.memory_events == 5
        assert metrics.compute_events == 2
        assert metrics.idle_time_ns == 200000
        assert metrics.utilization_pct == 80.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = StreamMetrics(
            stream_type="compute",
            total_events=10,
            total_time_ns=1000000,
            avg_event_time_ns=100000,
            max_event_time_ns=200000,
            min_event_time_ns=50000,
            sync_events=3,
            memory_events=5,
            compute_events=2,
            idle_time_ns=200000,
            utilization_pct=80.0
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["stream_type"] == "compute"
        assert result["total_events"] == 10
        assert result["total_time_ns"] == 1000000
        assert result["avg_event_time_ns"] == 100000
        assert result["max_event_time_ns"] == 200000
        assert result["min_event_time_ns"] == 50000
        assert result["sync_events"] == 3
        assert result["memory_events"] == 5
        assert result["compute_events"] == 2
        assert result["idle_time_ns"] == 200000
        assert result["utilization_pct"] == 80.0
    
    def test_calculate_utilization(self):
        """Test utilization calculation."""
        # High utilization
        metrics1 = StreamMetrics(
            stream_type="compute",
            total_events=10,
            total_time_ns=800000,
            avg_event_time_ns=80000,
            max_event_time_ns=100000,
            min_event_time_ns=60000,
            sync_events=2,
            memory_events=3,
            compute_events=5,
            idle_time_ns=200000,
            utilization_pct=0.0  # Will be calculated
        )
        
        # Calculate utilization (total_time / (total_time + idle_time))
        utilization1 = metrics1._calculate_utilization(1000000)  # 1ms total period
        assert utilization1 == 80.0  # 800000 / 1000000 * 100
        
        # Low utilization
        metrics2 = StreamMetrics(
            stream_type="copy",
            total_events=5,
            total_time_ns=200000,
            avg_event_time_ns=40000,
            max_event_time_ns=50000,
            min_event_time_ns=30000,
            sync_events=1,
            memory_events=4,
            compute_events=0,
            idle_time_ns=800000,
            utilization_pct=0.0  # Will be calculated
        )
        
        utilization2 = metrics2._calculate_utilization(1000000)  # 1ms total period
        assert utilization2 == 20.0  # 200000 / 1000000 * 100


class TestDualStreamManager:
    """Tests for DualStreamManager class."""
    
    def test_initialization(self):
        """Test stream manager initialization."""
        manager = DualStreamManager()
        
        # Check that streams are created
        assert hasattr(manager, 'compute_stream')
        assert hasattr(manager, 'copy_stream')
        assert hasattr(manager, 'events')
        assert hasattr(manager, 'metrics')
        
        # Check initial state
        assert manager.compute_stream is not None
        assert manager.copy_stream is not None
        assert isinstance(manager.events, list)
        assert len(manager.events) == 0
        
        # Check metrics
        assert "compute" in manager.metrics
        assert "copy" in manager.metrics
        assert isinstance(manager.metrics["compute"], StreamMetrics)
        assert isinstance(manager.metrics["copy"], StreamMetrics)
    
    def test_create_mock_stream(self):
        """Test mock stream creation."""
        manager = DualStreamManager()
        
        stream = manager._create_mock_stream("test_stream")
        
        assert isinstance(stream, dict)
        assert stream["stream_id"] == "test_stream"
        assert stream["stream_type"] == "test_stream"
        assert "created_at" in stream
        assert stream["is_active"] is True
    
    def test_create_mock_event(self):
        """Test mock event creation."""
        manager = DualStreamManager()
        
        event = manager._create_mock_event(
            event_type="compute",
            stream_type="compute",
            duration_ns=100000,
            description="Test computation"
        )
        
        assert isinstance(event, StreamEvent)
        assert event.event_type == "compute"
        assert event.stream_type == "compute"
        assert event.duration_ns == 100000
        assert "Test computation" in event.description
    
    def test_record_event(self):
        """Test event recording."""
        manager = DualStreamManager()
        
        initial_event_count = len(manager.events)
        
        # Record a compute event
        manager.record_event(
            event_type="compute",
            stream_type="compute",
            duration_ns=150000,
            description="Forward pass computation"
        )
        
        # Check that event was recorded
        assert len(manager.events) == initial_event_count + 1
        
        event = manager.events[-1]
        assert event.event_type == "compute"
        assert event.stream_type == "compute"
        assert event.duration_ns == 150000
        assert "Forward pass" in event.description
        
        # Check metrics were updated
        assert manager.metrics["compute"].total_events == 1
        assert manager.metrics["compute"].compute_events == 1
        assert manager.metrics["compute"].total_time_ns == 150000
    
    def test_synchronize_streams(self):
        """Test stream synchronization."""
        manager = DualStreamManager()
        
        # Record initial sync events
        initial_sync_events = sum(
            1 for e in manager.events if e.event_type == "sync"
        )
        
        # Synchronize streams
        sync_event = manager.synchronize_streams(
            description="Test synchronization"
        )
        
        # Check that sync event was created
        assert sync_event is not None
        assert sync_event.event_type == "sync"
        assert "Test synchronization" in sync_event.description
        
        # Check that event was recorded
        assert len(manager.events) > 0
        sync_events = [e for e in manager.events if e.event_type == "sync"]
        assert len(sync_events) == initial_sync_events + 1
        
        # Check metrics were updated for both streams
        assert manager.metrics["compute"].sync_events > 0
        assert manager.metrics["copy"].sync_events > 0
    
    def test_execute_compute_operation(self):
        """Test compute operation execution."""
        manager = DualStreamManager()
        
        # Record initial compute events
        initial_compute_events = manager.metrics["compute"].compute_events
        
        # Execute compute operation
        result = manager.execute_compute_operation(
            operation_type="attention",
            duration_ns=200000,
            description="Attention computation"
        )
        
        # Check result
        assert result is not None
        assert result["stream_type"] == "compute"
        assert result["operation_type"] == "attention"
        assert result["duration_ns"] == 200000
        assert result["success"] is True
        
        # Check that event was recorded
        assert manager.metrics["compute"].compute_events == initial_compute_events + 1
        assert manager.metrics["compute"].total_time_ns >= 200000
        
        # Check that a compute event was added
        compute_events = [e for e in manager.events if e.event_type == "compute"]
        assert len(compute_events) > 0
        last_compute = compute_events[-1]
        assert last_compute.stream_type == "compute"
        assert "Attention" in last_compute.description
    
    def test_execute_memory_operation(self):
        """Test memory operation execution."""
        manager = DualStreamManager()
        
        # Record initial memory events
        initial_memory_events = manager.metrics["copy"].memory_events
        
        # Execute memory operation
        result = manager.execute_memory_operation(
            operation_type="d2h",  # Device to host
            size_bytes=1024 * 1024,  # 1MB
            duration_ns=50000,
            description="Weight transfer to device"
        )
        
        # Check result
        assert result is not None
        assert result["stream_type"] == "copy"
        assert result["operation_type"] == "d2h"
        assert result["size_bytes"] == 1024 * 1024
        assert result["duration_ns"] == 50000
        assert result["success"] is True
        
        # Check that event was recorded
        assert manager.metrics["copy"].memory_events == initial_memory_events + 1
        assert manager.metrics["copy"].total_time_ns >= 50000
        
        # Check that a memory event was added
        memory_events = [e for e in manager.events if e.event_type == "memory"]
        assert len(memory_events) > 0
        last_memory = memory_events[-1]
        assert last_memory.stream_type == "copy"
        assert "Weight transfer" in last_memory.description
    
    def test_measure_overlap(self):
        """Test overlap measurement."""
        manager = DualStreamManager()
        
        # Execute some operations on both streams
        manager.execute_compute_operation(
            operation_type="layer1",
            duration_ns=300000,
            description="Layer 1 forward pass"
        )
        
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=512 * 1024,
            duration_ns=200000,
            description="Prefetch layer 2 weights"
        )
        
        # Synchronize
        manager.synchronize_streams(description="Sync after layer 1")
        
        # Execute more operations
        manager.execute_compute_operation(
            operation_type="layer2",
            duration_ns=280000,
            description="Layer 2 forward pass"
        )
        
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=512 * 1024,
            duration_ns=180000,
            description="Prefetch layer 3 weights"
        )
        
        # Measure overlap
        overlap_result = manager.measure_overlap()
        
        assert isinstance(overlap_result, dict)
        assert "total_time_ns" in overlap_result
        assert "compute_time_ns" in overlap_result
        assert "copy_time_ns" in overlap_result
        assert "overlap_time_ns" in overlap_result
        assert "overlap_percentage" in overlap_result
        assert "potential_overlap_ns" in overlap_result
        assert "potential_overlap_pct" in overlap_result
        
        # Check that values are reasonable
        assert overlap_result["total_time_ns"] > 0
        assert overlap_result["compute_time_ns"] > 0
        assert overlap_result["copy_time_ns"] > 0
        assert 0 <= overlap_result["overlap_percentage"] <= 100
        assert 0 <= overlap_result["potential_overlap_pct"] <= 100
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        manager = DualStreamManager()
        
        # Execute some operations
        manager.execute_compute_operation(
            operation_type="test",
            duration_ns=100000,
            description="Test compute"
        )
        
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=1024,
            duration_ns=50000,
            description="Test memory"
        )
        
        # Get metrics
        metrics = manager.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "compute" in metrics
        assert "copy" in metrics
        assert "overall" in metrics
        
        # Check stream metrics
        compute_metrics = metrics["compute"]
        copy_metrics = metrics["copy"]
        
        assert isinstance(compute_metrics, dict)
        assert isinstance(copy_metrics, dict)
        
        assert compute_metrics["stream_type"] == "compute"
        assert compute_metrics["total_events"] > 0
        assert compute_metrics["total_time_ns"] > 0
        
        assert copy_metrics["stream_type"] == "copy"
        assert copy_metrics["total_events"] > 0
        assert copy_metrics["total_time_ns"] > 0
        
        # Check overall metrics
        overall = metrics["overall"]
        assert "total_events" in overall
        assert "total_time_ns" in overall
        assert "utilization_pct" in overall
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        manager = DualStreamManager()
        
        # Execute some operations to populate metrics
        manager.execute_compute_operation(
            operation_type="test",
            duration_ns=100000,
            description="Test"
        )
        
        # Get initial metrics
        initial_metrics = manager.get_metrics()
        initial_compute_events = manager.metrics["compute"].total_events
        initial_events_count = len(manager.events)
        
        # Reset metrics
        manager.reset_metrics()
        
        # Get metrics after reset
        reset_metrics = manager.get_metrics()
        reset_compute_events = manager.metrics["compute"].total_events
        reset_events_count = len(manager.events)
        
        # Check that metrics were reset
        assert reset_compute_events == 0
        assert reset_events_count == 0
        
        # Check that get_metrics shows reset values
        assert reset_metrics["compute"]["total_events"] == 0
        assert reset_metrics["compute"]["total_time_ns"] == 0
        assert reset_metrics["overall"]["total_events"] == 0
    
    def test_save_events(self, tmp_path):
        """Test saving events to file."""
        manager = DualStreamManager()
        
        # Execute some operations
        manager.execute_compute_operation(
            operation_type="layer1",
            duration_ns=150000,
            description="Layer 1"
        )
        
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=1024 * 1024,
            duration_ns=80000,
            description="Weight transfer"
        )
        
        # Save events
        output_path = tmp_path / "test_events.json"
        saved_path = manager.save_events(output_path)
        
        assert saved_path.exists()
        
        # Load and verify
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert "events" in saved_data
        assert "metadata" in saved_data
        assert len(saved_data["events"]) == 2
        
        # Check event data
        for event_data in saved_data["events"]:
            assert "event_id" in event_data
            assert "event_type" in event_data
            assert "stream_type" in event_data
            assert "timestamp_ns" in event_data
            assert "duration_ns" in event_data
    
    def test_generate_report(self):
        """Test report generation."""
        manager = DualStreamManager()
        
        # Execute some operations
        manager.execute_compute_operation(
            operation_type="attention",
            duration_ns=200000,
            description="Attention block"
        )
        
        manager.execute_memory_operation(
            operation_type="d2h",
            size_bytes=512 * 1024,
            duration_ns=100000,
            description="Activation transfer"
        )
        
        manager.synchronize_streams(description="End of layer")
        
        # Generate report
        report = manager.generate_report()
        
        assert isinstance(report, str)
        assert "DUAL STREAM MANAGER REPORT" in report
        assert "WARNING" in report
        assert "SIMULATED" in report
        assert "compute" in report.lower()
        assert "copy" in report.lower()
        assert "utilization" in report.lower()
        assert "overlap" in report.lower()
    
    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        manager = DualStreamManager()
        
        # Execute some operations
        manager.execute_compute_operation(
            operation_type="test",
            duration_ns=100000,
            description="Test operation"
        )
        
        # Save report
        output_path = tmp_path / "test_report.txt"
        saved_path = manager.save_report(output_path)
        
        assert saved_path.exists()
        
        # Load and verify
        with open(saved_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "DUAL STREAM MANAGER REPORT" in report_content
        assert "WARNING" in report_content
        assert "SIMULATED" in report_content


class TestIntegration:
    """Integration tests for dual stream manager."""
    
    def test_concurrent_operations(self):
        """Test concurrent operations on both streams."""
        manager = DualStreamManager()
        
        # Simulate concurrent execution
        results = []
        
        # Start compute operation
        compute_result = manager.execute_compute_operation(
            operation_type="layer_forward",
            duration_ns=250000,
            description="Layer forward pass"
        )
        results.append(compute_result)
        
        # Start memory operation (should overlap with compute)
        memory_result = manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=1024 * 1024,
            duration_ns=200000,
            description="Next layer weight prefetch"
        )
        results.append(memory_result)
        
        # Synchronize
        sync_result = manager.synchronize_streams(
            description="Sync after concurrent operations"
        )
        
        # Check results
        assert len(results) == 2
        assert all(r["success"] is True for r in results)
        
        # Measure overlap
        overlap = manager.measure_overlap()
        assert overlap["overlap_percentage"] > 0  # Should have some overlap
        
        # Get metrics
        metrics = manager.get_metrics()
        assert metrics["compute"]["total_events"] == 1
        assert metrics["copy"]["total_events"] == 1
        assert metrics["overall"]["total_events"] == 2
    
    def test_full_workflow(self, tmp_path):
        """Test full workflow from initialization to reporting."""
        # Initialize manager
        manager = DualStreamManager()
        
        # Simulate a complete inference step
        # 1. Prefetch weights for layer 1
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=512 * 1024,
            duration_ns=80000,
            description="Prefetch layer 1 weights"
        )
        
        # 2. Compute layer 1
        manager.execute_compute_operation(
            operation_type="layer1",
            duration_ns=200000,
            description="Layer 1 forward pass"
        )
        
        # 3. Prefetch weights for layer 2 (overlaps with layer 1 compute)
        manager.execute_memory_operation(
            operation_type="h2d",
            size_bytes=512 * 1024,
            duration_ns=70000,
            description="Prefetch layer 2 weights"
        )
        
        # 4. Synchronize
        manager.synchronize_streams(description="Sync before layer 2")
        
        # 5. Compute layer 2
        manager.execute_compute_operation(
            operation_type="layer2",
            duration_ns=180000,
            description="Layer 2 forward pass"
        )
        
        # 6. Save results
        events_path = tmp_path / "workflow_events.json"
        report_path = tmp_path / "workflow_report.txt"
        
        manager.save_events(events_path)
        manager.save_report(report_path)
        
        # Verify files
        assert events_path.exists()
        assert report_path.exists()
        
        # Verify events file
        with open(events_path, 'r', encoding='utf-8') as f:
            events_data = json.load(f)
        
        assert len(events_data["events"]) == 5  # 2 memory + 2 compute + 1 sync
        
        # Verify report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "layer1" in report_content
        assert "layer2" in report_content
        assert "overlap" in report_content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])