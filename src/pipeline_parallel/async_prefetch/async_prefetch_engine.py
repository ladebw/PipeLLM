"""
WARNING: SIMULATED DATA — not from real hardware
These numbers are estimates/mocks and must be replaced with real 
measurements before any benchmarks are published (see ROADMAP tasks 
1.6, 2.7, 3.9, 4.2-4.5).

Async Double-Buffered Weight Prefetch Engine
Phase 2, Task 2.2: Set up dual CUDA stream infrastructure (compute + copy)

This module integrates dual CUDA streams with pinned memory buffers
to implement async double-buffered weight prefetch for LLM inference.

IMPORTANT: This engine uses simulated timing and CPU fallback.
All performance claims require validation with actual CUDA hardware
and real model workloads.
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import warnings
import threading

from .dual_stream_manager import DualStreamManager, StreamType
from .pinned_memory_pool import PinnedMemoryPool, MemoryBuffer, BufferState


class PrefetchState(Enum):
    """State of the prefetch engine."""
    IDLE = "idle"
    PREFETCHING = "prefetching"
    COMPUTING = "computing"
    SWAPPING = "swapping"
    ERROR = "error"


@dataclass
class WeightBuffer:
    """Double-buffered weight container."""
    buffer_id: int
    current_buffer: Optional[MemoryBuffer] = None
    next_buffer: Optional[MemoryBuffer] = None
    size_bytes: int = 0
    is_active: bool = False
    last_used: float = field(default_factory=time.time)
    
    def swap_buffers(self):
        """Swap current and next buffers."""
        self.current_buffer, self.next_buffer = self.next_buffer, self.current_buffer
        self.last_used = time.time()
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state information."""
        return {
            "buffer_id": self.buffer_id,
            "size_bytes": self.size_bytes,
            "is_active": self.is_active,
            "current_buffer_state": self.current_buffer.state.value if self.current_buffer else None,
            "next_buffer_state": self.next_buffer.state.value if self.next_buffer else None,
            "idle_seconds": time.time() - self.last_used,
        }


class AsyncPrefetchEngine:
    """
    Engine for async double-buffered weight prefetch.
    
    This class integrates dual CUDA streams with pinned memory buffers
    to enable overlapping of compute and memory transfer operations.
    """
    
    def __init__(self, 
                 memory_pool_size_mb: int = 2048,  # 2GB pool
                 enable_stats: bool = True,
                 device: str = "cuda"):
        """
        Initialize the async prefetch engine.
        
        Args:
            memory_pool_size_mb: Size of pinned memory pool in MB
            enable_stats: Whether to enable statistics collection
            device: Device to use (cuda or cpu)
        """
        self.device = device
        self.enable_stats = enable_stats
        
        # Initialize components
        self.stream_manager = DualStreamManager(device=device, enable_timing=True)
        self.memory_pool = PinnedMemoryPool(
            total_size_mb=memory_pool_size_mb,
            max_buffer_size_mb=512,
            min_buffer_size_mb=1,
            enable_stats=enable_stats
        )
        
        # State management
        self.state: PrefetchState = PrefetchState.IDLE
        self.weight_buffers: Dict[int, WeightBuffer] = {}
        self.next_buffer_id = 0
        
        # Statistics
        self.stats: Dict[str, Any] = {
            "total_prefetches": 0,
            "successful_prefetches": 0,
            "failed_prefetches": 0,
            "total_compute_time_ms": 0.0,
            "total_transfer_time_ms": 0.0,
            "overlapped_time_ms": 0.0,
            "average_speedup": 1.0,
            "buffer_hits": 0,
            "buffer_misses": 0,
            "peak_memory_usage_bytes": 0,
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Callbacks for monitoring
        self.state_change_callbacks: List[Callable] = []
        
        print(f"Async prefetch engine initialized:")
        print(f"  Device: {device}")
        print(f"  Memory pool: {memory_pool_size_mb} MB")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    
    def register_weight_buffer(self, size_bytes: int, metadata: Optional[Dict] = None) -> int:
        """
        Register a weight buffer for double-buffered prefetch.
        
        Args:
            size_bytes: Size of weight buffer in bytes
            metadata: Optional metadata for the buffer
            
        Returns:
            Buffer ID for future reference
        """
        with self.lock:
            buffer_id = self.next_buffer_id
            self.next_buffer_id += 1
            
            # Create weight buffer
            weight_buffer = WeightBuffer(
                buffer_id=buffer_id,
                size_bytes=size_bytes
            )
            
            self.weight_buffers[buffer_id] = weight_buffer
            
            # Allocate initial buffer
            current_buffer = self.memory_pool.allocate(size_bytes, metadata)
            if current_buffer:
                weight_buffer.current_buffer = current_buffer
                weight_buffer.is_active = True
                
                # Update statistics
                self._update_peak_memory()
                
                print(f"Registered weight buffer {buffer_id}: {size_bytes // (1024*1024)} MB")
            else:
                warnings.warn(f"Failed to allocate initial buffer for weight buffer {buffer_id}")
                weight_buffer.is_active = False
            
            return buffer_id
    
    def prefetch_next_weights(self, buffer_id: int, weight_data: torch.Tensor) -> bool:
        """
        Prefetch next weights for a buffer asynchronously.
        
        Args:
            buffer_id: ID of the weight buffer
            weight_data: Weight data to prefetch
            
        Returns:
            True if prefetch started successfully, False otherwise
        """
        with self.lock:
            if buffer_id not in self.weight_buffers:
                warnings.warn(f"Unknown buffer ID: {buffer_id}")
                return False
            
            weight_buffer = self.weight_buffers[buffer_id]
            if not weight_buffer.is_active:
                warnings.warn(f"Buffer {buffer_id} is not active")
                return False
            
            # Check if we already have a next buffer being prefetched
            if weight_buffer.next_buffer is not None:
                warnings.warn(f"Buffer {buffer_id} already has a next buffer being prefetched")
                return False
            
            # Allocate next buffer
            size_bytes = weight_buffer.size_bytes
            next_buffer = self.memory_pool.allocate(size_bytes, {"buffer_id": buffer_id})
            
            if not next_buffer:
                warnings.warn(f"Failed to allocate next buffer for {buffer_id}")
                return False
            
            # Copy weight data to buffer (CPU side)
            if next_buffer.cpu_ptr is not None:
                # Reshape and copy data
                # Note: This is simplified - in practice we'd need to handle tensor shapes
                try:
                    # Ensure we have enough space
                    required_elements = weight_data.numel()
                    available_elements = next_buffer.cpu_ptr.numel()
                    
                    if required_elements * weight_data.element_size() > size_bytes:
                        warnings.warn(f"Weight data too large for buffer {buffer_id}")
                        self.memory_pool.release(next_buffer)
                        return False
                    
                    # Copy data (simplified - would need proper tensor handling)
                    # next_buffer.cpu_ptr[:required_elements] = weight_data.flatten().cpu()
                    weight_buffer.next_buffer = next_buffer
                    
                    # Start async prefetch to GPU
                    self._start_async_prefetch(weight_buffer)
                    
                    # Update statistics
                    self.stats["total_prefetches"] += 1
                    self._update_peak_memory()
                    
                    return True
                    
                except Exception as e:
                    warnings.warn(f"Failed to copy weight data to buffer {buffer_id}: {e}")
                    self.memory_pool.release(next_buffer)
                    return False
            
            return False
    
    def _start_async_prefetch(self, weight_buffer: WeightBuffer):
        """Start async prefetch of next buffer to GPU."""
        if weight_buffer.next_buffer is None:
            return
        
        # Change state
        self._set_state(PrefetchState.PREFETCHING)
        
        # Use copy stream for async prefetch
        def prefetch_func():
            return self.memory_pool.prefetch_to_gpu(weight_buffer.next_buffer, non_blocking=True)
        
        # Execute on copy stream
        self.stream_manager.execute_copy_operation(prefetch_func)
        
        # Record start time for statistics
        weight_buffer.next_buffer.metadata["prefetch_start"] = time.time()
    
    def swap_buffers(self, buffer_id: int) -> bool:
        """
        Swap current and next buffers.
        
        Args:
            buffer_id: ID of the weight buffer
            
        Returns:
            True if swap successful, False otherwise
        """
        with self.lock:
            if buffer_id not in self.weight_buffers:
                warnings.warn(f"Unknown buffer ID: {buffer_id}")
                return False
            
            weight_buffer = self.weight_buffers[buffer_id]
            
            # Check if next buffer is ready
            if weight_buffer.next_buffer is None:
                warnings.warn(f"No next buffer ready for swap on buffer {buffer_id}")
                return False
            
            if weight_buffer.next_buffer.state != BufferState.IN_USE_ON_GPU:
                warnings.warn(f"Next buffer not on GPU for buffer {buffer_id}")
                return False
            
            # Change state
            self._set_state(PrefetchState.SWAPPING)
            
            # Wait for any pending operations
            self.stream_manager.synchronize_stream(StreamType.COPY)
            
            # Swap buffers
            old_buffer = weight_buffer.current_buffer
            weight_buffer.swap_buffers()
            
            # Release old buffer if it exists
            if old_buffer:
                self.memory_pool.release(old_buffer)
            
            # Update statistics
            if "prefetch_start" in weight_buffer.current_buffer.metadata:
                prefetch_time = time.time() - weight_buffer.current_buffer.metadata["prefetch_start"]
                self.stats["total_transfer_time_ms"] += prefetch_time * 1000
                self.stats["successful_prefetches"] += 1
            
            self._set_state(PrefetchState.COMPUTING)
            
            return True
    
    def execute_compute_with_prefetch(self, 
                                     buffer_id: int,
                                     compute_func: Callable,
                                     compute_args=(),
                                     compute_kwargs=None,
                                     next_weights: Optional[torch.Tensor] = None) -> Any:
        """
        Execute compute operation with async weight prefetch.
        
        Args:
            buffer_id: ID of the weight buffer to use
            compute_func: Compute function to execute
            compute_args: Arguments for compute function
            compute_kwargs: Keyword arguments for compute function
            next_weights: Next weights to prefetch (optional)
            
        Returns:
            Result of compute operation
        """
        with self.lock:
            compute_kwargs = compute_kwargs or {}
            
            # Start timing
            start_time = time.time()
            
            # Start prefetch of next weights if provided
            if next_weights is not None:
                self.prefetch_next_weights(buffer_id, next_weights)
            
            # Execute compute on compute stream
            def wrapped_compute():
                return compute_func(*compute_args, **compute_kwargs)
            
            compute_result = self.stream_manager.execute_compute_operation(wrapped_compute)
            
            # Wait for compute to complete
            self.stream_manager.synchronize_stream(StreamType.COMPUTE)
            
            # Calculate compute time
            compute_time = (time.time() - start_time) * 1000  # ms
            self.stats["total_compute_time_ms"] += compute_time
            
            # Swap buffers if next weights were prefetched
            if next_weights is not None:
                self.swap_buffers(buffer_id)
                
                # Calculate overlapped time
                # This is simplified - in practice we'd measure actual overlap
                prefetch_time = self.stats.get("last_prefetch_time_ms", 10.0)  # Example
                overlapped = min(compute_time, prefetch_time)
                self.stats["overlapped_time_ms"] += overlapped
                
                # Update speedup
                sequential_time = compute_time + prefetch_time
                overlapped_time = max(compute_time, prefetch_time)
                if overlapped_time > 0:
                    speedup = sequential_time / overlapped_time
                    # Update moving average
                    current_avg = self.stats["average_speedup"]
                    self.stats["average_speedup"] = current_avg * 0.9 + speedup * 0.1
            
            self._set_state(PrefetchState.IDLE)
            
            return compute_result
    
    def _set_state(self, new_state: PrefetchState):
        """Set engine state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        
        # Notify callbacks
        for callback in self.state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                warnings.warn(f"Error in state change callback: {e}")
    
    def _update_peak_memory(self):
        """Update peak memory usage statistic."""
        pool_info = self.memory_pool.get_pool_info()
        current_usage = pool_info["current_usage_bytes"]
        
        if current_usage > self.stats["peak_memory_usage_bytes"]:
            self.stats["peak_memory_usage_bytes"] = current_usage
    
    def register_state_callback(self, callback: Callable):
        """
        Register a callback for state changes.
        
        Args:
            callback: Function that takes (old_state, new_state)
        """
        self.state_change_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add stream manager stats
        stream_stats = self.stream_manager.get_metrics()
        stats.update({f"stream_{k}": v for k, v in stream_stats.items()})
        
        # Add memory pool stats
        pool_info = self.memory_pool.get_pool_info()
        if "statistics" in pool_info:
            stats.update({f"pool_{k}": v for k, v in pool_info["statistics"].items()})
        
        # Calculate derived statistics
        if stats["total_prefetches"] > 0:
            stats["prefetch_success_rate"] = (
                stats["successful_prefetches"] / stats["total_prefetches"] * 100
            )
        
        if stats["total_compute_time_ms"] > 0 and stats["total_transfer_time_ms"] > 0:
            stats["compute_transfer_ratio"] = (
                stats["total_compute_time_ms"] / stats["total_transfer_time_ms"]
            )
        
        if stats["overlapped_time_ms"] > 0 and stats["total_transfer_time_ms"] > 0:
            stats["overlap_efficiency"] = (
                stats["overlapped_time_ms"] / stats["total_transfer_time_ms"] * 100
            )
        
        return stats
    
    def get_buffer_info(self, buffer_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a weight buffer."""
        if buffer_id not in self.weight_buffers:
            return None
        
        weight_buffer = self.weight_buffers[buffer_id]
        info = weight_buffer.get_state()
        
        if weight_buffer.current_buffer:
            info["current_buffer_info"] = self.memory_pool.get_buffer_info(
                weight_buffer.current_buffer.buffer_id
            )
        
        if weight_buffer.next_buffer:
            info["next_buffer_info"] = self.memory_pool.get_buffer_info(
                weight_buffer.next_buffer.buffer_id
            )
        
        return info
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information."""
        return {
            "device": self.device,
            "state": self.state.value,
            "total_buffers": len(self.weight_buffers),
            "active_buffers": sum(1 for b in self.weight_buffers.values() if b.is_active),
            "stream_manager_info": self.stream_manager.generate_report(),
            "memory_pool_info": self.memory_pool.get_pool_info(),
            "statistics": self.get_stats(),
        }
    
    def cleanup(self):
        """Clean up all resources."""
        with self.lock:
            # Release all buffers
            for weight_buffer in self.weight_buffers.values():
                if weight_buffer.current_buffer:
                    self.memory_pool.release(weight_buffer.current_buffer)
                if weight_buffer.next_buffer:
                    self.memory_pool.release(weight_buffer.next_buffer)
            
            self.weight_buffers.clear()
            
            # Cleanup memory pool
            self.memory_pool.cleanup()
            
            print("Async prefetch engine cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        
        if exc_type is not None:
            print(f"Error in async prefetch engine context: {exc_val}")
        
        return False  # Don't suppress exceptions


# Example usage and demonstration
def demonstrate_async_prefetch():
    """Demonstrate async double-buffered weight prefetch."""
    print("=" * 80)
    print("DEMONSTRATING ASYNC DOUBLE-BUFFERED WEIGHT PREFETCH")
    print("=" * 80)
    
    # Create async prefetch engine
    with AsyncPrefetchEngine(memory_pool_size_mb=1024,  # 1GB pool
                            enable_stats=True,
                            device="cuda") as engine:
        
        # Get initial engine info
        engine_info = engine.get_engine_info()
        print(f"\nEngine initialized:")
        print(f"  Device: {engine_info['device']}")
        print(f"  State: {engine_info['state']}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        # Register weight buffers
        print("\nRegistering weight buffers...")
        buffer_sizes_mb = [50, 100, 200]  # Different weight sizes
        buffer_ids = []
        
        for size_mb in buffer_sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            buffer_id = engine.register_weight_buffer(
                size_bytes,
                metadata={"size_mb": size_mb, "purpose": "weight_buffer"}
            )
            buffer_ids.append(buffer_id)
            print(f"  Registered buffer {buffer_id}: {size_mb} MB")
        
        # Define mock compute function
        def mock_compute_operation(buffer_id: int, data_size: int = 1024):
            """Mock compute operation using weights."""
            # Simulate compute time based on buffer size
            buffer_info = engine.get_buffer_info(buffer_id)
            if buffer_info:
                size_mb = buffer_info.get("size_bytes", 0) // (1024 * 1024)
                compute_time = size_mb * 0.01  # 0.01ms per MB
                time.sleep(compute_time / 1000)  # Convert to seconds
            
            # Return mock result
            return torch.randn(data_size, data_size)
        
        # Define mock weight data generator
        def generate_mock_weights(size_mb: int) -> torch.Tensor:
            """Generate mock weight data."""
            elements = size_mb * 1024 * 1024 // 4  # float32: 4 bytes per element
            return torch.randn(elements)
        
        # Run demonstration
        print("\nRunning async prefetch demonstration...")
        
        # First compute without prefetch (baseline)
        print("\n1. Baseline (no prefetch):")
        start_time = time.time()
        
        # Generate weights for first buffer
        weights1 = generate_mock_weights(50)
        
        # Execute compute
        result1 = engine.execute_compute_with_prefetch(
            buffer_id=buffer_ids[0],
            compute_func=mock_compute_operation,
            compute_args=(buffer_ids[0], 1024),
            next_weights=None  # No prefetch
        )
        
        baseline_time = (time.time() - start_time) * 1000
        print(f"  Time: {baseline_time:.2f} ms")
        
        # Second compute with prefetch
        print("\n2. With async prefetch:")
        start_time = time.time()
        
        # Generate next weights while computing
        weights2 = generate_mock_weights(50)
        
        # Execute compute with prefetch
        result2 = engine.execute_compute_with_prefetch(
            buffer_id=buffer_ids[0],
            compute_func=mock_compute_operation,
            compute_args=(buffer_ids[0], 1024),
            next_weights=weights2  # Prefetch next weights
        )
        
        prefetch_time = (time.time() - start_time) * 1000
        print(f"  Time: {prefetch_time:.2f} ms")
        
        # Calculate improvement
        if baseline_time > 0:
            speedup = baseline_time / prefetch_time
            improvement = (1 - prefetch_time / baseline_time) * 100
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")
        
        # Third compute with continuous prefetch
        print("\n3. Continuous prefetch pipeline:")
        
        # Pipeline multiple operations
        operations = 5
        total_start = time.time()
        
        for i in range(operations):
            # Generate next weights
            next_weights = generate_mock_weights(50)
            
            # Execute with prefetch
            result = engine.execute_compute_with_prefetch(
                buffer_id=buffer_ids[0],
                compute_func=mock_compute_operation,
                compute_args=(buffer_ids[0], 1024),
                next_weights=next_weights if i < operations - 1 else None
            )
            
            print(f"  Operation {i+1} completed")
        
        pipeline_time = (time.time() - total_start) * 1000
        avg_time = pipeline_time / operations
        print(f"  Total pipeline time: {pipeline_time:.2f} ms")
        print(f"  Average per operation: {avg_time:.2f} ms")
        
        # Get final statistics
        stats = engine.get_stats()
        print(f"\nEngine statistics:")
        print(f"  Total prefetches: {stats['total_prefetches']}")
        print(f"  Successful prefetches: {stats['successful_prefetches']}")
        print(f"  Total compute time: {stats['total_compute_time_ms']:.2f} ms")
        print(f"  Total transfer time: {stats['total_transfer_time_ms']:.2f} ms")
        print(f"  Overlapped time: {stats['overlapped_time_ms']:.2f} ms")
        print(f"  Average speedup: {stats['average_speedup']:.2f}x")
        
        if 'overlap_efficiency' in stats:
            print(f"  Overlap efficiency: {stats['overlap_efficiency']:.1f}%")
        
        # Get buffer information
        print(f"\nBuffer information:")
        for buffer_id in buffer_ids:
            buffer_info = engine.get_buffer_info(buffer_id)
            if buffer_info:
                print(f"  Buffer {buffer_id}: {buffer_info['size_bytes'] // (1024*1024)} MB, "
                      f"Active: {buffer_info['is_active']}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_async_prefetch()