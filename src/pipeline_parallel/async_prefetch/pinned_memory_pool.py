"""
WARNING: SIMULATED DATA — not from real hardware
These numbers are estimates/mocks and must be replaced with real 
measurements before any benchmarks are published (see ROADMAP tasks 
1.6, 2.7, 3.9, 4.2-4.5).

Pinned Memory Buffer Pool for Weight Staging
Phase 2, Task 2.2: Set up dual CUDA stream infrastructure (compute + copy)

This module implements a pinned memory buffer pool for efficient weight
staging and transfer between CPU and GPU. Pinned (page-locked) memory
enables faster DMA transfers and is essential for async double-buffered
weight prefetch.

IMPORTANT: This implementation simulates pinned memory behavior.
Actual pinned memory performance requires CUDA hardware validation.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import time
import warnings
from collections import deque
import threading


class BufferState(Enum):
    """State of a memory buffer."""
    FREE = "free"
    ALLOCATED = "allocated"
    PENDING_COPY_TO_GPU = "pending_copy_to_gpu"
    IN_USE_ON_GPU = "in_use_on_gpu"
    PENDING_COPY_TO_CPU = "pending_copy_to_cpu"
    DIRTY = "dirty"  # Needs to be written back to CPU


@dataclass
class MemoryBuffer:
    """A pinned memory buffer for weight staging."""
    buffer_id: int
    size_bytes: int
    cpu_ptr: Optional[torch.Tensor] = None
    gpu_ptr: Optional[torch.Tensor] = None
    state: BufferState = BufferState.FREE
    allocation_time: float = field(default_factory=time.time)
    last_used_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_pinned(self) -> bool:
        """Check if the CPU buffer is pinned memory."""
        if self.cpu_ptr is None:
            return False
        return self.cpu_ptr.is_pinned()
    
    def to_gpu(self, non_blocking: bool = True) -> bool:
        """
        Copy buffer from CPU to GPU.
        
        Args:
            non_blocking: Whether to use non-blocking copy
            
        Returns:
            True if successful, False otherwise
        """
        if self.cpu_ptr is None:
            return False
        
        try:
            self.gpu_ptr = self.cpu_ptr.to('cuda', non_blocking=non_blocking)
            self.state = BufferState.IN_USE_ON_GPU
            self.last_used_time = time.time()
            return True
        except Exception as e:
            warnings.warn(f"Failed to copy buffer {self.buffer_id} to GPU: {e}")
            return False
    
    def to_cpu(self, non_blocking: bool = True) -> bool:
        """
        Copy buffer from GPU to CPU.
        
        Args:
            non_blocking: Whether to use non-blocking copy
            
        Returns:
            True if successful, False otherwise
        """
        if self.gpu_ptr is None:
            return False
        
        try:
            self.cpu_ptr = self.gpu_ptr.to('cpu', non_blocking=non_blocking)
            self.state = BufferState.FREE
            self.last_used_time = time.time()
            return True
        except Exception as e:
            warnings.warn(f"Failed to copy buffer {self.buffer_id} to CPU: {e}")
            return False
    
    def free(self):
        """Free the buffer memory."""
        self.cpu_ptr = None
        self.gpu_ptr = None
        self.state = BufferState.FREE
        self.metadata.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get buffer information."""
        return {
            "buffer_id": self.buffer_id,
            "size_bytes": self.size_bytes,
            "state": self.state.value,
            "is_pinned": self.is_pinned(),
            "allocation_time": self.allocation_time,
            "last_used_time": self.last_used_time,
            "age_seconds": time.time() - self.allocation_time,
            "idle_seconds": time.time() - self.last_used_time,
            "metadata_keys": list(self.metadata.keys()),
        }


class PinnedMemoryPool:
    """
    Pool of pinned memory buffers for efficient weight staging.
    
    This class manages a pool of pinned (page-locked) memory buffers
    that can be used for async weight prefetch between CPU and GPU.
    """
    
    def __init__(self, 
                 total_size_mb: int = 1024,  # 1GB default
                 max_buffer_size_mb: int = 256,  # 256MB max per buffer
                 min_buffer_size_mb: int = 1,  # 1MB min per buffer
                 enable_stats: bool = True):
        """
        Initialize the pinned memory pool.
        
        Args:
            total_size_mb: Total pool size in megabytes
            max_buffer_size_mb: Maximum buffer size in megabytes
            min_buffer_size_mb: Minimum buffer size in megabytes
            enable_stats: Whether to enable statistics collection
        """
        self.total_size_bytes = total_size_mb * 1024 * 1024
        self.max_buffer_size_bytes = max_buffer_size_mb * 1024 * 1024
        self.min_buffer_size_bytes = min_buffer_size_mb * 1024 * 1024
        
        # Buffer management
        self.buffers: Dict[int, MemoryBuffer] = {}
        self.free_buffers: List[int] = []  # Buffer IDs of free buffers
        self.allocated_buffers: Dict[int, MemoryBuffer] = {}  # Currently allocated buffers
        self.next_buffer_id = 0
        
        # Statistics
        self.enable_stats = enable_stats
        self.stats: Dict[str, Any] = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "total_bytes_allocated": 0,
            "peak_memory_usage_bytes": 0,
            "current_memory_usage_bytes": 0,
            "allocation_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_allocation_time_ms": 0.0,
            "total_transfer_time_ms": 0.0,
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize pool
        self._initialize_pool()
        
        print(f"Pinned memory pool initialized:")
        print(f"  Total size: {total_size_mb} MB")
        print(f"  Max buffer size: {max_buffer_size_mb} MB")
        print(f"  Min buffer size: {min_buffer_size_mb} MB")
        print(f"  Initial buffers: {len(self.buffers)}")
    
    def _initialize_pool(self):
        """Initialize the memory pool with some buffers."""
        # Create initial buffers of various sizes
        buffer_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128]
        
        total_allocated = 0
        for size_mb in buffer_sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            
            # Check if we have enough space
            if total_allocated + size_bytes > self.total_size_bytes:
                break
            
            # Create buffer
            buffer = self._create_buffer(size_bytes)
            if buffer is not None:
                total_allocated += size_bytes
                self.free_buffers.append(buffer.buffer_id)
    
    def _create_buffer(self, size_bytes: int) -> Optional[MemoryBuffer]:
        """
        Create a new pinned memory buffer.
        
        Args:
            size_bytes: Size of buffer in bytes
            
        Returns:
            MemoryBuffer if successful, None otherwise
        """
        if size_bytes > self.max_buffer_size_bytes:
            warnings.warn(f"Buffer size {size_bytes} exceeds maximum {self.max_buffer_size_bytes}")
            return None
        
        if size_bytes < self.min_buffer_size_bytes:
            warnings.warn(f"Buffer size {size_bytes} below minimum {self.min_buffer_size_bytes}")
            return None
        
        buffer_id = self.next_buffer_id
        self.next_buffer_id += 1
        
        try:
            # Allocate pinned memory on CPU
            # Note: torch.empty_pinned is not available in all versions
            # Using workaround with pin_memory=True
            cpu_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, 
                                    pin_memory=True)
            
            buffer = MemoryBuffer(
                buffer_id=buffer_id,
                size_bytes=size_bytes,
                cpu_ptr=cpu_tensor,
                state=BufferState.FREE
            )
            
            self.buffers[buffer_id] = buffer
            
            # Update statistics
            if self.enable_stats:
                self.stats["total_allocations"] += 1
                self.stats["total_bytes_allocated"] += size_bytes
                self.stats["current_memory_usage_bytes"] += size_bytes
                self.stats["peak_memory_usage_bytes"] = max(
                    self.stats["peak_memory_usage_bytes"],
                    self.stats["current_memory_usage_bytes"]
                )
            
            return buffer
            
        except Exception as e:
            warnings.warn(f"Failed to create pinned buffer of size {size_bytes}: {e}")
            
            if self.enable_stats:
                self.stats["allocation_failures"] += 1
            
            return None
    
    def allocate(self, size_bytes: int, metadata: Optional[Dict] = None) -> Optional[MemoryBuffer]:
        """
        Allocate a buffer of the specified size.
        
        Args:
            size_bytes: Size needed in bytes
            metadata: Optional metadata to attach to buffer
            
        Returns:
            Allocated buffer, or None if allocation failed
        """
        with self.lock:
            start_time = time.time()
            
            # First, try to find a free buffer of appropriate size
            best_buffer_id = None
            best_size_diff = float('inf')
            
            for buffer_id in self.free_buffers:
                buffer = self.buffers[buffer_id]
                
                # Check if buffer is large enough
                if buffer.size_bytes >= size_bytes:
                    size_diff = buffer.size_bytes - size_bytes
                    
                    # Prefer buffer that fits best (smallest excess)
                    if size_diff < best_size_diff:
                        best_size_diff = size_diff
                        best_buffer_id = buffer_id
            
            if best_buffer_id is not None:
                # Found a suitable buffer in pool
                buffer = self.buffers[best_buffer_id]
                self.free_buffers.remove(best_buffer_id)
                self.allocated_buffers[buffer_id] = buffer
                
                buffer.state = BufferState.ALLOCATED
                buffer.last_used_time = time.time()
                if metadata:
                    buffer.metadata.update(metadata)
                
                if self.enable_stats:
                    self.stats["cache_hits"] += 1
                    allocation_time = (time.time() - start_time) * 1000
                    # Update moving average of allocation time
                    current_avg = self.stats["average_allocation_time_ms"]
                    self.stats["average_allocation_time_ms"] = (
                        current_avg * 0.9 + allocation_time * 0.1
                    )
                
                return buffer
            
            # No suitable buffer found, need to allocate new one
            if self.enable_stats:
                self.stats["cache_misses"] += 1
            
            # Check if we have space for new allocation
            current_usage = self.stats["current_memory_usage_bytes"]
            if current_usage + size_bytes > self.total_size_bytes:
                # Try to free some memory
                if not self._free_unused_buffers(size_bytes):
                    warnings.warn(f"Insufficient memory in pool. Need {size_bytes}, "
                                f"available {self.total_size_bytes - current_usage}")
                    return None
            
            # Create new buffer
            buffer = self._create_buffer(size_bytes)
            if buffer is None:
                return None
            
            buffer.state = BufferState.ALLOCATED
            buffer.last_used_time = time.time()
            if metadata:
                buffer.metadata.update(metadata)
            
            self.allocated_buffers[buffer.buffer_id] = buffer
            
            if self.enable_stats:
                allocation_time = (time.time() - start_time) * 1000
                current_avg = self.stats["average_allocation_time_ms"]
                self.stats["average_allocation_time_ms"] = (
                    current_avg * 0.9 + allocation_time * 0.1
                )
            
            return buffer
    
    def _free_unused_buffers(self, required_bytes: int) -> bool:
        """
        Free unused buffers to make space.
        
        Args:
            required_bytes: Bytes needed
            
        Returns:
            True if enough space was freed, False otherwise
        """
        # Sort free buffers by size (largest first)
        free_buffers_sorted = sorted(
            [(self.buffers[buffer_id], buffer_id) for buffer_id in self.free_buffers],
            key=lambda x: x[0].size_bytes,
            reverse=True
        )
        
        bytes_freed = 0
        buffers_to_remove = []
        
        for buffer, buffer_id in free_buffers_sorted:
            if bytes_freed >= required_bytes:
                break
            
            # Free this buffer
            buffer.free()
            buffers_to_remove.append(buffer_id)
            bytes_freed += buffer.size_bytes
            
            if self.enable_stats:
                self.stats["total_deallocations"] += 1
                self.stats["current_memory_usage_bytes"] -= buffer.size_bytes
        
        # Remove freed buffers from pool
        for buffer_id in buffers_to_remove:
            self.free_buffers.remove(buffer_id)
            del self.buffers[buffer_id]
        
        return bytes_freed >= required_bytes
    
    def release(self, buffer: MemoryBuffer):
        """
        Release a buffer back to the pool.
        
        Args:
            buffer: Buffer to release
        """
        with self.lock:
            if buffer.buffer_id not in self.allocated_buffers:
                warnings.warn(f"Buffer {buffer.buffer_id} not in allocated buffers")
                return
            
            # Reset buffer state
            buffer.state = BufferState.FREE
            buffer.last_used_time = time.time()
            buffer.metadata.clear()
            
            # Move from allocated to free
            del self.allocated_buffers[buffer.buffer_id]
            self.free_buffers.append(buffer.buffer_id)
    
    def prefetch_to_gpu(self, buffer: MemoryBuffer, non_blocking: bool = True) -> bool:
        """
        Prefetch buffer to GPU asynchronously.
        
        Args:
            buffer: Buffer to prefetch
            non_blocking: Whether to use non-blocking copy
            
        Returns:
            True if successful, False otherwise
        """
        if buffer.state != BufferState.ALLOCATED:
            warnings.warn(f"Cannot prefetch buffer in state {buffer.state}")
            return False
        
        start_time = time.time()
        success = buffer.to_gpu(non_blocking=non_blocking)
        
        if self.enable_stats and success:
            transfer_time = (time.time() - start_time) * 1000
            self.stats["total_transfer_time_ms"] += transfer_time
        
        return success
    
    def writeback_to_cpu(self, buffer: MemoryBuffer, non_blocking: bool = True) -> bool:
        """
        Write buffer back to CPU asynchronously.
        
        Args:
            buffer: Buffer to write back
            non_blocking: Whether to use non-blocking copy
            
        Returns:
            True if successful, False otherwise
        """
        if buffer.state != BufferState.IN_USE_ON_GPU:
            warnings.warn(f"Cannot write back buffer in state {buffer.state}")
            return False
        
        start_time = time.time()
        success = buffer.to_cpu(non_blocking=non_blocking)
        
        if self.enable_stats and success:
            transfer_time = (time.time() - start_time) * 1000
            self.stats["total_transfer_time_ms"] += transfer_time
        
        return success
    
    def get_buffer_info(self, buffer_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific buffer."""
        if buffer_id not in self.buffers:
            return None
        return self.buffers[buffer_id].get_info()
    
    def get_pool_info(self) -> Dict[str, Any]:
        """Get information about the entire pool."""
        total_buffers = len(self.buffers)
        free_buffers = len(self.free_buffers)
        allocated_buffers = len(self.allocated_buffers)
        
        # Calculate size distribution
        size_distribution = {}
        for buffer in self.buffers.values():
            size_mb = buffer.size_bytes // (1024 * 1024)
            size_distribution[size_mb] = size_distribution.get(size_mb, 0) + 1
        
        info = {
            "total_buffers": total_buffers,
            "free_buffers": free_buffers,
            "allocated_buffers": allocated_buffers,
            "utilization_percentage": (allocated_buffers / total_buffers * 100) if total_buffers > 0 else 0,
            "size_distribution_mb": size_distribution,
            "total_size_bytes": self.total_size_bytes,
            "current_usage_bytes": self.stats["current_memory_usage_bytes"],
            "usage_percentage": (self.stats["current_memory_usage_bytes"] / self.total_size_bytes * 100) 
                              if self.total_size_bytes > 0 else 0,
        }
        
        if self.enable_stats:
            info["statistics"] = self.stats.copy()
        
        return info
    
    def cleanup(self, max_idle_seconds: float = 300):
        """
        Clean up idle buffers.
        
        Args:
            max_idle_seconds: Maximum idle time before buffer is cleaned up
        """
        with self.lock:
            current_time = time.time()
            buffers_to_remove = []
            
            for buffer_id in self.free_buffers[:]:  # Copy list for safe iteration
                buffer = self.buffers[buffer_id]
                idle_time = current_time - buffer.last_used_time
                
                if idle_time > max_idle_seconds:
                    buffers_to_remove.append(buffer_id)
            
            # Remove idle buffers
            for buffer_id in buffers_to_remove:
                buffer = self.buffers[buffer_id]
                buffer.free()
                self.free_buffers.remove(buffer_id)
                del self.buffers[buffer_id]
                
                if self.enable_stats:
                    self.stats["total_deallocations"] += 1
                    self.stats["current_memory_usage_bytes"] -= buffer.size_bytes
            
            if buffers_to_remove:
                print(f"Cleaned up {len(buffers_to_remove)} idle buffers")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        
        if exc_type is not None:
            print(f"Error in pinned memory pool context: {exc_val}")
        
        return False  # Don't suppress exceptions


# Example usage and demonstration
def demonstrate_pinned_memory_pool():
    """Demonstrate the pinned memory pool."""
    print("=" * 80)
    print("DEMONSTRATING PINNED MEMORY POOL")
    print("=" * 80)
    
    # Create pinned memory pool
    with PinnedMemoryPool(total_size_mb=512,  # 512MB pool
                         max_buffer_size_mb=128,
                         min_buffer_size_mb=1,
                         enable_stats=True) as pool:
        
        # Get initial pool info
        pool_info = pool.get_pool_info()
        print(f"\nInitial pool state:")
        print(f"  Total buffers: {pool_info['total_buffers']}")
        print(f"  Free buffers: {pool_info['free_buffers']}")
        print(f"  Allocated buffers: {pool_info['allocated_buffers']}")
        print(f"  Utilization: {pool_info['utilization_percentage']:.1f}%")
        print(f"  Memory usage: {pool_info['usage_percentage']:.1f}%")
        
        # Allocate some buffers
        print("\nAllocating buffers...")
        buffer_sizes = [10 * 1024 * 1024,  # 10MB
                       50 * 1024 * 1024,  # 50MB
                       100 * 1024 * 1024]  # 100MB
        
        allocated_buffers = []
        
        for i, size_bytes in enumerate(buffer_sizes):
            buffer = pool.allocate(size_bytes, metadata={"purpose": f"test_buffer_{i}"})
            if buffer:
                allocated_buffers.append(buffer)
                print(f"  Allocated buffer {buffer.buffer_id}: {size_bytes // (1024*1024)} MB")
            else:
                print(f"  Failed to allocate {size_bytes // (1024*1024)} MB buffer")
        
        # Get pool info after allocation
        pool_info = pool.get_pool_info()
        print(f"\nAfter allocation:")
        print(f"  Free buffers: {pool_info['free_buffers']}")
        print(f"  Allocated buffers: {pool_info['allocated_buffers']}")
        print(f"  Utilization: {pool_info['utilization_percentage']:.1f}%")
        
        # Demonstrate prefetch to GPU
        print("\nPrefetching buffers to GPU...")
        for buffer in allocated_buffers:
            if pool.prefetch_to_gpu(buffer, non_blocking=True):
                print(f"  Prefetched buffer {buffer.buffer_id} to GPU")
            else:
                print(f"  Failed to prefetch buffer {buffer.buffer_id}")
        
        # Wait a bit (simulating compute)
        print("\nSimulating compute operations...")
        time.sleep(0.5)
        
        # Write back to CPU
        print("\nWriting buffers back to CPU...")
        for buffer in allocated_buffers:
            if pool.writeback_to_cpu(buffer, non_blocking=True):
                print(f"  Wrote back buffer {buffer.buffer_id} to CPU")
            else:
                print(f"  Failed to write back buffer {buffer.buffer_id}")
        
        # Release buffers
        print("\nReleasing buffers...")
        for buffer in allocated_buffers:
            pool.release(buffer)
            print(f"  Released buffer {buffer.buffer_id}")
        
        # Get final pool info
        pool_info = pool.get_pool_info()
        print(f"\nFinal pool state:")
        print(f"  Free buffers: {pool_info['free_buffers']}")
        print(f"  Allocated buffers: {pool_info['allocated_buffers']}")
        print(f"  Utilization: {pool_info['utilization_percentage']:.1f}%")
        
        # Show statistics
        if "statistics" in pool_info:
            stats = pool_info["statistics"]
            print(f"\nStatistics:")
            print(f"  Total allocations: {stats['total_allocations']}")
            print(f"  Cache hits: {stats['cache_hits']}")
            print(f"  Cache misses: {stats['cache_misses']}")
            print(f"  Hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100:.1f}%")
            print(f"  Average allocation time: {stats['average_allocation_time_ms']:.2f} ms")
            print(f"  Total transfer time: {stats['total_transfer_time_ms']:.2f} ms")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_pinned_memory_pool()