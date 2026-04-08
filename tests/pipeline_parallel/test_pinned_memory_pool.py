#!/usr/bin/env python3
"""
Tests for pinned memory pool (Task 2.3).

WARNING: SIMULATED DATA — not from real hardware
These tests verify the memory pool infrastructure, not actual CUDA performance.
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

# Import pinned memory pool
from src.pipeline_parallel.async_prefetch.pinned_memory_pool import (
    PinnedMemoryPool,
    MemoryBuffer,
    MemoryPoolMetrics
)


class TestMemoryBuffer:
    """Tests for MemoryBuffer class."""
    
    def test_creation(self):
        """Test creating a MemoryBuffer instance."""
        buffer = MemoryBuffer(
            buffer_id="test_buffer_1",
            size_bytes=1024 * 1024,  # 1MB
            buffer_type="weight",
            is_pinned=True,
            is_allocated=True,
            allocation_time_ns=1000000,
            last_used_time_ns=1500000,
            usage_count=5,
            stream_id="compute_stream_1"
        )
        
        assert buffer.buffer_id == "test_buffer_1"
        assert buffer.size_bytes == 1024 * 1024
        assert buffer.buffer_type == "weight"
        assert buffer.is_pinned is True
        assert buffer.is_allocated is True
        assert buffer.allocation_time_ns == 1000000
        assert buffer.last_used_time_ns == 1500000
        assert buffer.usage_count == 5
        assert buffer.stream_id == "compute_stream_1"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        buffer = MemoryBuffer(
            buffer_id="test_buffer_1",
            size_bytes=1024 * 1024,
            buffer_type="weight",
            is_pinned=True,
            is_allocated=True,
            allocation_time_ns=1000000,
            last_used_time_ns=1500000,
            usage_count=5,
            stream_id="compute_stream_1"
        )
        
        result = buffer.to_dict()
        
        assert isinstance(result, dict)
        assert result["buffer_id"] == "test_buffer_1"
        assert result["size_bytes"] == 1024 * 1024
        assert result["buffer_type"] == "weight"
        assert result["is_pinned"] is True
        assert result["is_allocated"] is True
        assert result["allocation_time_ns"] == 1000000
        assert result["last_used_time_ns"] == 1500000
        assert result["usage_count"] == 5
        assert result["stream_id"] == "compute_stream_1"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "buffer_id": "test_buffer_2",
            "size_bytes": 512 * 1024,
            "buffer_type": "activation",
            "is_pinned": False,
            "is_allocated": True,
            "allocation_time_ns": 2000000,
            "last_used_time_ns": 2500000,
            "usage_count": 3,
            "stream_id": "copy_stream_1"
        }
        
        buffer = MemoryBuffer.from_dict(data)
        
        assert buffer.buffer_id == "test_buffer_2"
        assert buffer.size_bytes == 512 * 1024
        assert buffer.buffer_type == "activation"
        assert buffer.is_pinned is False
        assert buffer.is_allocated is True
        assert buffer.allocation_time_ns == 2000000
        assert buffer.last_used_time_ns == 2500000
        assert buffer.usage_count == 3
        assert buffer.stream_id == "copy_stream_1"
    
    def test_update_usage(self):
        """Test updating buffer usage."""
        buffer = MemoryBuffer(
            buffer_id="test_buffer",
            size_bytes=1024 * 1024,
            buffer_type="weight",
            is_pinned=True,
            is_allocated=True,
            allocation_time_ns=1000000,
            last_used_time_ns=1000000,
            usage_count=0,
            stream_id="compute_stream"
        )
        
        initial_usage_count = buffer.usage_count
        initial_last_used = buffer.last_used_time_ns
        
        # Update usage
        buffer.update_usage(stream_id="new_stream")
        
        assert buffer.usage_count == initial_usage_count + 1
        assert buffer.last_used_time_ns > initial_last_used
        assert buffer.stream_id == "new_stream"
    
    def test_release(self):
        """Test releasing buffer."""
        buffer = MemoryBuffer(
            buffer_id="test_buffer",
            size_bytes=1024 * 1024,
            buffer_type="weight",
            is_pinned=True,
            is_allocated=True,
            allocation_time_ns=1000000,
            last_used_time_ns=1500000,
            usage_count=5,
            stream_id="compute_stream"
        )
        
        assert buffer.is_allocated is True
        
        # Release buffer
        buffer.release()
        
        assert buffer.is_allocated is False
        assert buffer.stream_id is None


class TestMemoryPoolMetrics:
    """Tests for MemoryPoolMetrics class."""
    
    def test_creation(self):
        """Test creating a MemoryPoolMetrics instance."""
        metrics = MemoryPoolMetrics(
            total_buffers=10,
            allocated_buffers=7,
            free_buffers=3,
            total_memory_bytes=100 * 1024 * 1024,  # 100MB
            allocated_memory_bytes=70 * 1024 * 1024,  # 70MB
            free_memory_bytes=30 * 1024 * 1024,  # 30MB
            allocation_requests=50,
            allocation_successes=45,
            allocation_failures=5,
            deallocation_requests=40,
            hit_count=35,
            miss_count=15,
            hit_rate=70.0,
            avg_allocation_time_ns=50000,
            max_allocation_time_ns=200000,
            min_allocation_time_ns=10000
        )
        
        assert metrics.total_buffers == 10
        assert metrics.allocated_buffers == 7
        assert metrics.free_buffers == 3
        assert metrics.total_memory_bytes == 100 * 1024 * 1024
        assert metrics.allocated_memory_bytes == 70 * 1024 * 1024
        assert metrics.free_memory_bytes == 30 * 1024 * 1024
        assert metrics.allocation_requests == 50
        assert metrics.allocation_successes == 45
        assert metrics.allocation_failures == 5
        assert metrics.deallocation_requests == 40
        assert metrics.hit_count == 35
        assert metrics.miss_count == 15
        assert metrics.hit_rate == 70.0
        assert metrics.avg_allocation_time_ns == 50000
        assert metrics.max_allocation_time_ns == 200000
        assert metrics.min_allocation_time_ns == 10000
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = MemoryPoolMetrics(
            total_buffers=10,
            allocated_buffers=7,
            free_buffers=3,
            total_memory_bytes=100 * 1024 * 1024,
            allocated_memory_bytes=70 * 1024 * 1024,
            free_memory_bytes=30 * 1024 * 1024,
            allocation_requests=50,
            allocation_successes=45,
            allocation_failures=5,
            deallocation_requests=40,
            hit_count=35,
            miss_count=15,
            hit_rate=70.0,
            avg_allocation_time_ns=50000,
            max_allocation_time_ns=200000,
            min_allocation_time_ns=10000
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["total_buffers"] == 10
        assert result["allocated_buffers"] == 7
        assert result["free_buffers"] == 3
        assert result["total_memory_bytes"] == 100 * 1024 * 1024
        assert result["allocated_memory_bytes"] == 70 * 1024 * 1024
        assert result["free_memory_bytes"] == 30 * 1024 * 1024
        assert result["allocation_requests"] == 50
        assert result["allocation_successes"] == 45
        assert result["allocation_failures"] == 5
        assert result["deallocation_requests"] == 40
        assert result["hit_count"] == 35
        assert result["miss_count"] == 15
        assert result["hit_rate"] == 70.0
        assert result["avg_allocation_time_ns"] == 50000
        assert result["max_allocation_time_ns"] == 200000
        assert result["min_allocation_time_ns"] == 10000
    
    def test_update_hit_rate(self):
        """Test hit rate calculation."""
        metrics = MemoryPoolMetrics(
            total_buffers=0,
            allocated_buffers=0,
            free_buffers=0,
            total_memory_bytes=0,
            allocated_memory_bytes=0,
            free_memory_bytes=0,
            allocation_requests=0,
            allocation_successes=0,
            allocation_failures=0,
            deallocation_requests=0,
            hit_count=30,
            miss_count=10,
            hit_rate=0.0,  # Will be calculated
            avg_allocation_time_ns=0,
            max_allocation_time_ns=0,
            min_allocation_time_ns=0
        )
        
        # Calculate hit rate
        hit_rate = metrics._calculate_hit_rate()
        assert hit_rate == 75.0  # 30 / (30 + 10) * 100
        
        # Test with zero hits
        metrics.hit_count = 0
        metrics.miss_count = 0
        hit_rate_zero = metrics._calculate_hit_rate()
        assert hit_rate_zero == 0.0


class TestPinnedMemoryPool:
    """Tests for PinnedMemoryPool class."""
    
    def test_initialization(self):
        """Test memory pool initialization."""
        pool = PinnedMemoryPool(
            max_memory_bytes=1024 * 1024 * 1024,  # 1GB
            buffer_sizes=[64 * 1024, 256 * 1024, 1024 * 1024],  # 64KB, 256KB, 1MB
            enable_pinning=True
        )
        
        assert pool.max_memory_bytes == 1024 * 1024 * 1024
        assert pool.buffer_sizes == [64 * 1024, 256 * 1024, 1024 * 1024]
        assert pool.enable_pinning is True
        assert hasattr(pool, 'buffers')
        assert hasattr(pool, 'metrics')
        assert isinstance(pool.buffers, dict)
        assert isinstance(pool.metrics, MemoryPoolMetrics)
        
        # Check that buffers were initialized
        assert len(pool.buffers) > 0
        
        # Check initial metrics
        assert pool.metrics.total_buffers > 0
        assert pool.metrics.free_buffers == pool.metrics.total_buffers
        assert pool.metrics.allocated_buffers == 0
    
    def test_create_mock_buffer(self):
        """Test mock buffer creation."""
        pool = PinnedMemoryPool()
        
        buffer = pool._create_mock_buffer(
            buffer_id="test_buffer",
            size_bytes=1024 * 1024,
            buffer_type="weight",
            is_pinned=True
        )
        
        assert isinstance(buffer, MemoryBuffer)
        assert buffer.buffer_id == "test_buffer"
        assert buffer.size_bytes == 1024 * 1024
        assert buffer.buffer_type == "weight"
        assert buffer.is_pinned is True
        assert buffer.is_allocated is False  # Not allocated yet
        assert buffer.usage_count == 0
    
    def test_initialize_buffers(self):
        """Test buffer initialization."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,  # 10MB
            buffer_sizes=[1024 * 1024, 2 * 1024 * 1024],  # 1MB, 2MB
            enable_pinning=True
        )
        
        # Check that buffers were created
        assert len(pool.buffers) > 0
        
        # Check buffer sizes
        total_memory = 0
        for buffer in pool.buffers.values():
            total_memory += buffer.size_bytes
            assert buffer.size_bytes in [1024 * 1024, 2 * 1024 * 1024]
            assert buffer.is_pinned is True
            assert buffer.is_allocated is False
        
        # Check that total memory doesn't exceed max
        assert total_memory <= 10 * 1024 * 1024
        
        # Check metrics
        assert pool.metrics.total_buffers == len(pool.buffers)
        assert pool.metrics.total_memory_bytes == total_memory
        assert pool.metrics.free_buffers == len(pool.buffers)
        assert pool.metrics.allocated_buffers == 0
    
    def test_allocate_buffer(self):
        """Test buffer allocation."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[1024 * 1024],
            enable_pinning=True
        )
        
        initial_allocated = pool.metrics.allocated_buffers
        initial_requests = pool.metrics.allocation_requests
        
        # Allocate a buffer
        buffer = pool.allocate_buffer(
            size_bytes=1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        assert buffer is not None
        assert isinstance(buffer, MemoryBuffer)
        assert buffer.size_bytes == 1024 * 1024
        assert buffer.buffer_type == "weight"
        assert buffer.is_allocated is True
        assert buffer.stream_id == "compute_stream"
        assert buffer.usage_count == 1
        
        # Check metrics
        assert pool.metrics.allocated_buffers == initial_allocated + 1
        assert pool.metrics.allocation_requests == initial_requests + 1
        assert pool.metrics.allocation_successes == initial_requests + 1
        assert pool.metrics.hit_count == 1  # Should be a hit (found suitable buffer)
    
    def test_allocate_buffer_too_large(self):
        """Test allocation of buffer that's too large."""
        pool = PinnedMemoryPool(
            max_memory_bytes=1 * 1024 * 1024,  # 1MB max
            buffer_sizes=[512 * 1024],  # 512KB buffers
            enable_pinning=True
        )
        
        initial_failures = pool.metrics.allocation_failures
        
        # Try to allocate buffer larger than any available
        buffer = pool.allocate_buffer(
            size_bytes=2 * 1024 * 1024,  # 2MB - too large
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        assert buffer is None  # Should fail
        
        # Check metrics
        assert pool.metrics.allocation_failures == initial_failures + 1
    
    def test_allocate_buffer_no_free(self):
        """Test allocation when no free buffers available."""
        pool = PinnedMemoryPool(
            max_memory_bytes=2 * 1024 * 1024,  # 2MB
            buffer_sizes=[1 * 1024 * 1024],  # 1MB buffers
            enable_pinning=True
        )
        
        # Allocate all buffers
        buffers = []
        for _ in range(2):  # Should have 2 buffers (2MB total / 1MB each)
            buffer = pool.allocate_buffer(
                size_bytes=1 * 1024 * 1024,
                buffer_type="weight",
                stream_id="compute_stream"
            )
            assert buffer is not None
            buffers.append(buffer)
        
        # Try to allocate another buffer (should fail)
        initial_failures = pool.metrics.allocation_failures
        buffer = pool.allocate_buffer(
            size_bytes=1 * 1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        assert buffer is None  # Should fail
        
        # Check metrics
        assert pool.metrics.allocation_failures == initial_failures + 1
    
    def test_release_buffer(self):
        """Test buffer release."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[1024 * 1024],
            enable_pinning=True
        )
        
        # Allocate a buffer
        buffer = pool.allocate_buffer(
            size_bytes=1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        assert buffer is not None
        assert buffer.is_allocated is True
        
        initial_allocated = pool.metrics.allocated_buffers
        initial_deallocations = pool.metrics.deallocation_requests
        
        # Release the buffer
        success = pool.release_buffer(buffer.buffer_id)
        
        assert success is True
        assert buffer.is_allocated is False
        assert buffer.stream_id is None
        
        # Check metrics
        assert pool.metrics.allocated_buffers == initial_allocated - 1
        assert pool.metrics.deallocation_requests == initial_deallocations + 1
    
    def test_release_nonexistent_buffer(self):
        """Test releasing a buffer that doesn't exist."""
        pool = PinnedMemoryPool()
        
        initial_deallocations = pool.metrics.deallocation_requests
        
        # Try to release non-existent buffer
        success = pool.release_buffer("nonexistent_buffer")
        
        assert success is False
        
        # Check metrics (deallocation count shouldn't change)
        assert pool.metrics.deallocation_requests == initial_deallocations
    
    def test_find_suitable_buffer(self):
        """Test finding suitable buffer."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[512 * 1024, 1024 * 1024, 2 * 1024 * 1024],
            enable_pinning=True
        )
        
        # Test finding exact size
        buffer_id = pool._find_suitable_buffer(1024 * 1024)
        assert buffer_id is not None
        
        buffer = pool.buffers[buffer_id]
        assert buffer.size_bytes == 1024 * 1024
        assert buffer.is_allocated is False
        
        # Test finding larger buffer (size not available)
        buffer_id = pool._find_suitable_buffer(768 * 1024)  # Between 512KB and 1MB
        assert buffer_id is not None
        
        buffer = pool.buffers[buffer_id]
        assert buffer.size_bytes == 1024 * 1024  # Should pick 1MB buffer
        assert buffer.is_allocated is False
        
        # Test when no suitable buffer exists
        # First allocate all large buffers
        large_buffers = []
        for buf_id, buf in pool.buffers.items():
            if buf.size_bytes == 2 * 1024 * 1024 and not buf.is_allocated:
                buf.is_allocated = True
                large_buffers.append(buf_id)
        
        # Now try to find 2MB buffer (should fail)
        buffer_id = pool._find_suitable_buffer(2 * 1024 * 1024)
        assert buffer_id is None  # All 2MB buffers are allocated
    
    def test_get_buffer_info(self):
        """Test getting buffer information."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[1024 * 1024],
            enable_pinning=True
        )
        
        # Allocate a buffer
        buffer = pool.allocate_buffer(
            size_bytes=1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        # Get buffer info
        info = pool.get_buffer_info(buffer.buffer_id)
        
        assert isinstance(info, dict)
        assert info["buffer_id"] == buffer.buffer_id
        assert info["size_bytes"] == buffer.size_bytes
        assert info["buffer_type"] == buffer.buffer_type
        assert info["is_allocated"] is True
        assert info["stream_id"] == "compute_stream"
        assert info["usage_count"] == 1
    
    def test_get_buffer_info_nonexistent(self):
        """Test getting info for non-existent buffer."""
        pool = PinnedMemoryPool()
        
        info = pool.get_buffer_info("nonexistent_buffer")
        
        assert info is None
    
    def test_get_pool_status(self):
        """Test getting pool status."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[1024 * 1024, 2 * 1024 * 1024],
            enable_pinning=True
        )
        
        # Allocate some buffers
        allocated_buffers = []
        for _ in range(2):
            buffer = pool.allocate_buffer(
                size_bytes=1024 * 1024,
                buffer_type="weight",
                stream_id="compute_stream"
            )
            if buffer:
                allocated_buffers.append(buffer)
        
        # Get pool status
        status = pool.get_pool_status()
        
        assert isinstance(status, dict)
        assert "total_buffers" in status
        assert "allocated_buffers" in status
        assert "free_buffers" in status
        assert "total_memory_bytes" in status
        assert "allocated_memory_bytes" in status
        assert "free_memory_bytes" in status
        assert "utilization_percentage" in status
        assert "hit_rate" in status
        
        # Check values
        assert status["total_buffers"] == pool.metrics.total_buffers
        assert status["allocated_buffers"] == len(allocated_buffers)
        assert status["free_buffers"] == status["total_buffers"] - len(allocated_buffers)
        assert status["utilization_percentage"] > 0
        assert 0 <= status["hit_rate"] <= 100
    
    def test_reset_pool(self):
        """Test resetting the pool."""
        pool = PinnedMemoryPool(
            max_memory_bytes=10 * 1024 * 1024,
            buffer_sizes=[1024 * 1024],
            enable_pinning=True
        )
        
        # Allocate some buffers
        buffers = []
        for _ in range(3):
            buffer = pool.allocate_buffer(
                size_bytes=1024 * 1024,
                buffer_type="weight",
                stream_id="compute_stream"
            )
            if buffer:
                buffers.append(buffer)
        
        # Check that buffers are allocated
        assert pool.metrics.allocated_buffers > 0
        
        # Reset pool
        pool.reset_pool()
        
        # Check that all buffers are free
        assert pool.metrics.allocated_buffers == 0
        assert pool.metrics.free_buffers == pool.metrics.total_buffers
        
        # Check individual buffers
        for buffer in pool.buffers.values():
            assert buffer.is_allocated is False
            assert buffer.stream_id is None
            assert buffer.usage_count == 0  # Usage count is reset
    
    def test_generate_report(self):
        """Test report generation."""
        pool = PinnedMemoryPool(
            max_memory_bytes=100 * 1024 * 1024,
            buffer_sizes=[16 * 1024 * 1024, 32 * 1024 * 1024],
            enable_pinning=True
        )
        
        # Perform some operations
        buffer1 = pool.allocate_buffer(
            size_bytes=16 * 1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        buffer2 = pool.allocate_buffer(
            size_bytes=32 * 1024 * 1024,
            buffer_type="activation",
            stream_id="copy_stream"
        )
        
        # Generate report
        report = pool.generate_report()
        
        assert isinstance(report, str)
        assert "PINNED MEMORY POOL REPORT" in report
        assert "WARNING" in report
        assert "SIMULATED" in report
        assert "total_buffers" in report.lower()
        assert "allocated" in report.lower()
        assert "free" in report.lower()
        assert "hit rate" in report.lower()
        
        # Check that buffer info is in report
        if buffer1:
            assert buffer1.buffer_id in report
        if buffer2:
            assert buffer2.buffer_id in report
    
    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        pool = PinnedMemoryPool()
        
        # Perform some operations
        pool.allocate_buffer(
            size_bytes=1024 * 1024,
            buffer_type="weight",
            stream_id="compute_stream"
        )
        
        # Save report
        output_path = tmp_path / "pool_report.txt"
        saved_path = pool.save_report(output_path)
        
        assert saved_path.exists()
        
        # Load and verify
        with open(saved_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "PINNED MEMORY POOL REPORT" in report_content
        assert "WARNING" in report_content
        assert "SIMULATED" in report_content


class TestIntegration:
    """Integration tests for pinned memory pool."""
    
    def test_buffer_lifecycle(self):
        """Test complete buffer lifecycle."""
        pool = PinnedMemoryPool(
            max_memory_bytes=50 * 1024 * 1024,  # 50MB
            buffer_sizes=[8 * 1024 * 1024, 16 * 1024 * 1024],  # 8MB, 16MB
            enable_pinning=True
        )
        
        # 1. Allocate buffers
        buffers = []
        for i in range(3):
            buffer = pool.allocate_buffer(
                size_bytes=8 * 1024 * 1024,
                buffer_type=f"weight_{i}",
                stream_id=f"stream_{i}"
            )
            assert buffer is not None
            buffers.append(buffer)
        
        # Check allocation
        assert pool.metrics.allocated_buffers == 3
        assert pool.metrics.allocation_successes == 3
        
        # 2. Use buffers (simulate usage)
        for buffer in buffers:
            buffer.update_usage(stream_id="compute_stream")
        
        # 3. Release buffers
        for buffer in buffers:
            success = pool.release_buffer(buffer.buffer_id)
            assert success is True
        
        # Check release
        assert pool.metrics.allocated_buffers == 0
        assert pool.metrics.deallocation_requests == 3
        
        # 4. Reallocate (should reuse buffers)
        reused_buffers = []
        for i in range(3):
            buffer = pool.allocate_buffer(
                size_bytes=8 * 1024 * 1024,
                buffer_type=f"weight_reused_{i}",
                stream_id=f"stream_{i}"
            )
            assert buffer is not None
            reused_buffers.append(buffer)
        
        # Check that buffers were reused (usage count > 1)
        for buffer in reused_buffers:
            assert buffer.usage_count > 1
    
    def test_memory_fragmentation_simulation(self):
        """Test memory fragmentation scenario."""
        pool = PinnedMemoryPool(
            max_memory_bytes=64 * 1024 * 1024,  # 64MB
            buffer_sizes=[8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024],
            enable_pinning=True
        )
        
        # Allocate all 32MB buffers (should have 2: 64MB / 32MB = 2)
        large_buffers = []
        for _ in range(2):
            buffer = pool.allocate_buffer(
                size_bytes=32 * 1024 * 1024,
                buffer_type="large_weight",
                stream_id="compute_stream"
            )
            assert buffer is not None
            large_buffers.append(buffer)
        
        # Try to allocate another 32MB buffer (should fail)
        buffer = pool.allocate_buffer(
            size_bytes=32 * 1024 * 1024,
            buffer_type="large_weight",
            stream_id="compute_stream"
        )
        assert buffer is None
        
        # But should be able to allocate smaller buffers
        small_buffer = pool.allocate_buffer(
            size_bytes=8 * 1024 * 1024,
            buffer_type="small_weight",
            stream_id="compute_stream"
        )
        assert small_buffer is not None
        
        # Release one large buffer
        pool.release_buffer(large_buffers[0].buffer_id)
        
        # Now should be able to allocate 32MB buffer again
        buffer = pool.allocate_buffer(
            size_bytes=32 * 1024 * 1024,
            buffer_type="large_weight",
            stream_id="compute_stream"
        )
        assert buffer is not None
    
    def test_concurrent_allocation_simulation(self):
        """Test simulated concurrent allocation patterns."""
        pool = PinnedMemoryPool(
            max_memory_bytes=128 * 1024 * 1024,  # 128MB
            buffer_sizes=[4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024],
            enable_pinning=True
        )
        
        # Simulate allocation pattern from multiple "streams"
        streams = ["compute_stream", "copy_stream", "prefetch_stream"]
        buffers_by_stream = {stream: [] for stream in streams}
        
        # Allocate buffers for each stream
        for stream in streams:
            for i in range(2):  # 2 buffers per stream
                buffer = pool.allocate_buffer(
                    size_bytes=8 * 1024 * 1024,
                    buffer_type=f"{stream}_buffer_{i}",
                    stream_id=stream
                )
                assert buffer is not None
                buffers_by_stream[stream].append(buffer)
        
        # Check allocation
        total_allocated = sum(len(buffers) for buffers in buffers_by_stream.values())
        assert pool.metrics.allocated_buffers == total_allocated
        
        # Simulate usage pattern
        for stream, buffers in buffers_by_stream.items():
            for buffer in buffers:
                # Simulate multiple uses
                for _ in range(3):
                    buffer.update_usage(stream_id=stream)
        
        # Release buffers from one stream
        stream_to_release = streams[0]
        for buffer in buffers_by_stream[stream_to_release]:
            success = pool.release_buffer(buffer.buffer_id)
            assert success is True
        
        # Check that buffers are free
        assert pool.metrics.free_buffers == len(buffers_by_stream[stream_to_release])
        
        # Generate report to verify everything
        report = pool.generate_report()
        assert "compute_stream" in report
        assert "copy_stream" in report
        assert "prefetch_stream" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])