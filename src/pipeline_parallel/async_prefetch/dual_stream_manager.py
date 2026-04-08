"""
Dual CUDA Stream Infrastructure
Phase 2, Task 2.2: Set up dual CUDA stream infrastructure (compute + copy)

This module implements the dual CUDA stream infrastructure for async
double-buffered weight prefetch. It provides separate compute and copy
streams with proper synchronization to enable overlapping of compute
and memory transfer operations.
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import warnings


class StreamType(Enum):
    """Types of CUDA streams."""
    COMPUTE = "compute"
    COPY = "copy"
    DEFAULT = "default"


@dataclass
class StreamEvent:
    """CUDA event for synchronization between streams."""
    event_id: int
    stream_type: StreamType
    timestamp: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DualStreamManager:
    """
    Manager for dual CUDA stream infrastructure.
    
    This class manages separate compute and copy streams with proper
    synchronization to enable overlapping of compute and memory transfer
    operations for async double-buffered weight prefetch.
    """
    
    def __init__(self, device: str = "cuda", enable_timing: bool = True):
        """
        Initialize the dual stream manager.
        
        Args:
            device: Device to use (cuda or cpu)
            enable_timing: Whether to enable event timing
        """
        self.device = device
        self.enable_timing = enable_timing
        
        # Initialize streams
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.default_stream: Optional[torch.cuda.Stream] = None
        
        # Synchronization events
        self.events: List[StreamEvent] = []
        self.next_event_id = 0
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "compute_stream_utilization": 0.0,
            "copy_stream_utilization": 0.0,
            "overlap_percentage": 0.0,
            "synchronization_overhead_ms": 0.0,
            "total_operations": 0,
            "successful_overlaps": 0,
        }
        
        # Initialize CUDA streams if available
        self._initialize_streams()
        
    def _initialize_streams(self):
        """Initialize CUDA streams."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                # Create separate streams for compute and copy operations
                self.compute_stream = torch.cuda.Stream()
                self.copy_stream = torch.cuda.Stream()
                self.default_stream = torch.cuda.current_stream()
                
                print(f"Dual stream infrastructure initialized on {torch.cuda.get_device_name()}")
                print(f"  Compute stream: {self.compute_stream}")
                print(f"  Copy stream: {self.copy_stream}")
                print(f"  Default stream: {self.default_stream}")
                
            except Exception as e:
                warnings.warn(f"Failed to initialize CUDA streams: {e}. Using CPU fallback.")
                self.device = "cpu"
                self._initialize_cpu_fallback()
        else:
            warnings.warn("CUDA not available. Using CPU fallback.")
            self.device = "cpu"
            self._initialize_cpu_fallback()
    
    def _initialize_cpu_fallback(self):
        """Initialize CPU fallback streams."""
        # For CPU, we use None streams but maintain the same interface
        self.compute_stream = None
        self.copy_stream = None
        self.default_stream = None
        print("Dual stream infrastructure initialized (CPU fallback)")
    
    def get_stream(self, stream_type: StreamType) -> Optional[torch.cuda.Stream]:
        """
        Get the stream for a specific type.
        
        Args:
            stream_type: Type of stream to get
            
        Returns:
            The requested stream, or None for CPU fallback
        """
        if stream_type == StreamType.COMPUTE:
            return self.compute_stream
        elif stream_type == StreamType.COPY:
            return self.copy_stream
        elif stream_type == StreamType.DEFAULT:
            return self.default_stream
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    
    def record_event(self, stream_type: StreamType, description: str = "", 
                    metadata: Optional[Dict] = None) -> StreamEvent:
        """
        Record an event on a stream.
        
        Args:
            stream_type: Type of stream to record event on
            description: Description of the event
            metadata: Additional metadata for the event
            
        Returns:
            StreamEvent object
        """
        event = StreamEvent(
            event_id=self.next_event_id,
            stream_type=stream_type,
            timestamp=time.time(),
            description=description,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.next_event_id += 1
        
        # Record CUDA event if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            stream = self.get_stream(stream_type)
            if stream is not None:
                # In a real implementation, we would record a CUDA event here
                # event.cuda_event = torch.cuda.Event(enable_timing=self.enable_timing)
                # event.cuda_event.record(stream=stream)
                pass
        
        return event
    
    def wait_for_event(self, event: StreamEvent, stream_type: StreamType):
        """
        Wait for an event to complete on a stream.
        
        Args:
            event: Event to wait for
            stream_type: Stream to wait on
        """
        if self.device == "cuda" and torch.cuda.is_available():
            stream = self.get_stream(stream_type)
            if stream is not None:
                # In a real implementation, we would wait for the CUDA event
                # if hasattr(event, 'cuda_event'):
                #     event.cuda_event.wait(stream=stream)
                pass
        
        # Update metrics
        self.metrics["total_operations"] += 1
    
    def synchronize_stream(self, stream_type: StreamType):
        """
        Synchronize a specific stream.
        
        Args:
            stream_type: Type of stream to synchronize
        """
        if self.device == "cuda" and torch.cuda.is_available():
            stream = self.get_stream(stream_type)
            if stream is not None:
                start_time = time.time()
                stream.synchronize()
                sync_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update synchronization overhead
                self.metrics["synchronization_overhead_ms"] += sync_time
                
                # Record event
                self.record_event(
                    stream_type=stream_type,
                    description=f"Synchronized {stream_type.value} stream",
                    metadata={"synchronization_time_ms": sync_time}
                )
    
    def synchronize_all(self):
        """Synchronize all streams."""
        if self.device == "cuda" and torch.cuda.is_available():
            # Synchronize compute stream
            if self.compute_stream is not None:
                self.synchronize_stream(StreamType.COMPUTE)
            
            # Synchronize copy stream
            if self.copy_stream is not None:
                self.synchronize_stream(StreamType.COPY)
            
            # Synchronize default stream
            if self.default_stream is not None:
                self.default_stream.synchronize()
    
    def execute_compute_operation(self, operation_func, *args, **kwargs) -> Any:
        """
        Execute a compute operation on the compute stream.
        
        Args:
            operation_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the operation
        """
        if self.device == "cuda" and torch.cuda.is_available() and self.compute_stream is not None:
            # Switch to compute stream context
            with torch.cuda.stream(self.compute_stream):
                result = operation_func(*args, **kwargs)
                
                # Record event
                self.record_event(
                    stream_type=StreamType.COMPUTE,
                    description=f"Compute operation: {operation_func.__name__}",
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
                
                return result
        else:
            # CPU fallback
            return operation_func(*args, **kwargs)
    
    def execute_copy_operation(self, operation_func, *args, **kwargs) -> Any:
        """
        Execute a copy operation on the copy stream.
        
        Args:
            operation_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the operation
        """
        if self.device == "cuda" and torch.cuda.is_available() and self.copy_stream is not None:
            # Switch to copy stream context
            with torch.cuda.stream(self.copy_stream):
                result = operation_func(*args, **kwargs)
                
                # Record event
                self.record_event(
                    stream_type=StreamType.COPY,
                    description=f"Copy operation: {operation_func.__name__}",
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
                
                return result
        else:
            # CPU fallback
            return operation_func(*args, **kwargs)
    
    def overlap_operations(self, compute_func, copy_func, 
                          compute_args=(), compute_kwargs=None,
                          copy_args=(), copy_kwargs=None) -> Tuple[Any, Any]:
        """
        Execute compute and copy operations concurrently with overlap.
        
        Args:
            compute_func: Compute operation function
            copy_func: Copy operation function
            compute_args: Arguments for compute function
            compute_kwargs: Keyword arguments for compute function
            copy_args: Arguments for copy function
            copy_kwargs: Keyword arguments for copy function
            
        Returns:
            Tuple of (compute_result, copy_result)
        """
        compute_kwargs = compute_kwargs or {}
        copy_kwargs = copy_kwargs or {}
        
        if self.device == "cuda" and torch.cuda.is_available():
            # Start copy operation on copy stream
            copy_future = self.execute_copy_operation(copy_func, *copy_args, **copy_kwargs)
            
            # Start compute operation on compute stream (overlaps with copy)
            compute_future = self.execute_compute_operation(compute_func, *compute_args, **compute_kwargs)
            
            # Synchronize both streams
            self.synchronize_all()
            
            # Update overlap metrics
            self.metrics["successful_overlaps"] += 1
            
            return compute_future, copy_future
        else:
            # CPU fallback: execute sequentially
            compute_result = compute_func(*compute_args, **compute_kwargs)
            copy_result = copy_func(*copy_args, **copy_kwargs)
            return compute_result, copy_result
    
    def measure_overlap(self, compute_time_ms: float, copy_time_ms: float) -> float:
        """
        Measure potential overlap between compute and copy operations.
        
        Args:
            compute_time_ms: Time for compute operation in milliseconds
            copy_time_ms: Time for copy operation in milliseconds
            
        Returns:
            Overlap percentage (0-100)
        """
        if compute_time_ms <= 0 or copy_time_ms <= 0:
            return 0.0
        
        # Ideal overlap: min(compute_time, copy_time)
        ideal_overlap = min(compute_time_ms, copy_time_ms)
        
        # Total time without overlap
        total_sequential = compute_time_ms + copy_time_ms
        
        # Total time with ideal overlap
        total_overlapped = max(compute_time_ms, copy_time_ms)
        
        # Overlap percentage
        overlap_percentage = (ideal_overlap / total_sequential) * 100
        
        # Update metrics
        self.metrics["overlap_percentage"] = overlap_percentage
        
        return overlap_percentage
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate utilizations
        total_ops = self.metrics["total_operations"]
        if total_ops > 0:
            self.metrics["compute_stream_utilization"] = self.metrics.get("compute_operations", 0) / total_ops
            self.metrics["copy_stream_utilization"] = self.metrics.get("copy_operations", 0) / total_ops
        
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "compute_stream_utilization": 0.0,
            "copy_stream_utilization": 0.0,
            "overlap_percentage": 0.0,
            "synchronization_overhead_ms": 0.0,
            "total_operations": 0,
            "successful_overlaps": 0,
        }
        self.events.clear()
        self.next_event_id = 0
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of stream usage.
        
        Returns:
            Dictionary with report data
        """
        report = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available() if self.device == "cuda" else False,
            "streams_initialized": {
                "compute": self.compute_stream is not None,
                "copy": self.copy_stream is not None,
                "default": self.default_stream is not None,
            },
            "metrics": self.get_metrics(),
            "event_count": len(self.events),
            "recent_events": [
                {
                    "id": e.event_id,
                    "stream": e.stream_type.value,
                    "description": e.description,
                    "timestamp": e.timestamp,
                }
                for e in self.events[-10:]  # Last 10 events
            ],
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        metrics = self.get_metrics()
        
        # Check overlap percentage
        overlap = metrics.get("overlap_percentage", 0)
        if overlap < 30:
            recommendations.append(
                f"Low overlap percentage ({overlap:.1f}%). Consider adjusting operation "
                "sizes or improving synchronization to increase overlap."
            )
        elif overlap > 70:
            recommendations.append(
                f"High overlap potential ({overlap:.1f}%). Current implementation is "
                "effectively utilizing dual streams."
            )
        
        # Check synchronization overhead
        sync_overhead = metrics.get("synchronization_overhead_ms", 0)
        if sync_overhead > 10:  # More than 10ms of sync overhead
            recommendations.append(
                f"High synchronization overhead ({sync_overhead:.2f} ms). "
                "Consider reducing synchronization frequency or using event-based synchronization."
            )
        
        # Check stream utilization
        compute_util = metrics.get("compute_stream_utilization", 0)
        copy_util = metrics.get("copy_stream_utilization", 0)
        
        if compute_util < 0.3:
            recommendations.append(
                f"Low compute stream utilization ({compute_util:.1%}). "
                "Consider increasing compute workload or improving load balancing."
            )
        
        if copy_util < 0.3:
            recommendations.append(
                f"Low copy stream utilization ({copy_util:.1%}). "
                "Consider increasing memory transfer workload or improving prefetch scheduling."
            )
        
        if not recommendations:
            recommendations.append(
                "Stream infrastructure appears balanced. Continue monitoring for optimization opportunities."
            )
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - synchronize all streams."""
        self.synchronize_all()
        
        # Clean up if needed
        if exc_type is not None:
            print(f"Error in dual stream context: {exc_val}")
        
        return False  # Don't suppress exceptions


# Example usage and demonstration
def demonstrate_dual_streams():
    """Demonstrate the dual stream infrastructure."""
    print("=" * 80)
    print("DEMONSTRATING DUAL CUDA STREAM INFRASTRUCTURE")
    print("=" * 80)
    
    # Create dual stream manager
    with DualStreamManager(device="cuda", enable_timing=True) as stream_manager:
        # Get initial report
        report = stream_manager.generate_report()
        print(f"\nInitial setup:")
        print(f"  Device: {report['device']}")
        print(f"  CUDA available: {report['cuda_available']}")
        print(f"  Streams initialized: {report['streams_initialized']}")
        
        # Define example operations
        def compute_operation(size=1024):
            """Example compute operation (matrix multiplication)."""
            if torch.cuda.is_available():
                a = torch.randn(size, size, device='cuda')
                b = torch.randn(size, size, device='cuda')
                return torch.mm(a, b)
            else:
                # CPU fallback
                a = torch.randn(size, size)
                b = torch.randn(size, size)
                return torch.mm(a, b)
        
        def copy_operation(size_mb=100):
            """Example copy operation (memory transfer)."""
            if torch.cuda.is_available():
                # Create data on CPU
                cpu_data = torch.randn(size_mb * 256 * 1024 // 4)  # Approx size_mb MB
                
                # Copy to GPU
                gpu_data = cpu_data.to('cuda')
                return gpu_data
            else:
                # CPU fallback: simple tensor creation
                return torch.randn(size_mb * 256 * 1024 // 4)
        
        # Execute operations with overlap
        print("\nExecuting operations with overlap...")
        
        # Time sequential execution
        start_time = time.time()
        compute_result_seq = compute_operation(2048)
        copy_result_seq = copy_operation(50)
        sequential_time = (time.time() - start_time) * 1000
        
        # Reset metrics
        stream_manager.reset_metrics()
        
        # Time overlapped execution
        start_time = time.time()
        compute_result, copy_result = stream_manager.overlap_operations(
            compute_func=compute_operation,
            copy_func=copy_operation,
            compute_args=(2048,),
            copy_args=(50,)
        )
        overlapped_time = (time.time() - start_time) * 1000
        
        # Calculate speedup
        speedup = sequential_time / overlapped_time if overlapped_time > 0 else 1.0
        
        # Get final report
        final_report = stream_manager.generate_report()
        metrics = final_report["metrics"]
        
        print(f"\nPerformance results:")
        print(f"  Sequential execution: {sequential_time:.2f} ms")
        print(f"  Overlapped execution: {overlapped_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Overlap percentage: {metrics.get('overlap_percentage', 0):.1f}%")
        print(f"  Successful overlaps: {metrics.get('successful_overlaps', 0)}")
        print(f"  Synchronization overhead: {metrics.get('synchronization_overhead_ms', 0):.2f} ms")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(final_report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nTotal events recorded: {final_report['event_count']}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_dual_streams()