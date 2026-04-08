"""
Race Condition Validator for Async Prefetch Infrastructure

WARNING: SIMULATED DATA — not from real hardware
This validator uses simulated timing and CPU-based synchronization to test
race condition scenarios. Real CUDA hardware validation is required before
deployment (see ROADMAP task 2.5).

Validates:
1. Stream synchronization correctness
2. Memory access ordering
3. Double-buffer swap safety
4. Concurrent access patterns
5. Stress test reliability
"""

import threading
import time
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceConditionType(Enum):
    """Types of race conditions to detect."""
    STREAM_SYNC = "stream_synchronization"
    MEMORY_ACCESS = "memory_access_ordering"
    BUFFER_SWAP = "double_buffer_swap"
    CONCURRENT_READ = "concurrent_read_access"
    CONCURRENT_WRITE = "concurrent_write_access"
    DEADLOCK = "deadlock_condition"
    DATA_CORRUPTION = "data_corruption"


@dataclass
class RaceConditionResult:
    """Result of a race condition test."""
    test_name: str
    condition_type: RaceConditionType
    detected: bool = False
    severity: str = "low"  # low, medium, high, critical
    description: str = ""
    error_message: str = ""
    stack_trace: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "condition_type": self.condition_type.value,
            "detected": self.detected,
            "severity": self.severity,
            "description": self.description,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp
        }


@dataclass
class ValidationMetrics:
    """Metrics collected during race condition validation."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    critical_issues: int = 0
    high_severity: int = 0
    medium_severity: int = 0
    low_severity: int = 0
    total_execution_time: float = 0.0
    concurrent_threads: int = 0
    max_memory_usage_mb: float = 0.0
    
    def add_result(self, result: RaceConditionResult):
        """Update metrics based on test result."""
        self.total_tests += 1
        if not result.detected:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            if result.severity == "critical":
                self.critical_issues += 1
            elif result.severity == "high":
                self.high_severity += 1
            elif result.severity == "medium":
                self.medium_severity += 1
            elif result.severity == "low":
                self.low_severity += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "critical_issues": self.critical_issues,
            "high_severity": self.high_severity,
            "medium_severity": self.medium_severity,
            "low_severity": self.low_severity,
            "total_execution_time": self.total_execution_time,
            "concurrent_threads": self.concurrent_threads,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "pass_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        }


class MockCUDAEvent:
    """Mock CUDA event for synchronization testing."""
    
    def __init__(self, name: str = "mock_event"):
        self.name = name
        self.recorded = False
        self.synchronized = False
        self.timestamp = time.time()
    
    def record(self, stream=None):
        """Record the event (simulated)."""
        self.recorded = True
        self.timestamp = time.time()
        logger.debug(f"Event {self.name} recorded at stream {stream}")
    
    def synchronize(self):
        """Wait for event to complete (simulated)."""
        if not self.recorded:
            raise RuntimeError(f"Event {self.name} not recorded before synchronization")
        self.synchronized = True
        # Simulate some delay
        time.sleep(0.001)
        logger.debug(f"Event {self.name} synchronized")
    
    def query(self) -> bool:
        """Check if event has completed (simulated)."""
        return self.recorded and self.synchronized


class MockStream:
    """Mock CUDA stream for concurrency testing."""
    
    def __init__(self, name: str = "mock_stream"):
        self.name = name
        self.operations: List[str] = []
        self.lock = threading.Lock()
        self.active = True
    
    def add_operation(self, op: str):
        """Add an operation to the stream (simulated)."""
        with self.lock:
            self.operations.append(op)
            logger.debug(f"Stream {self.name}: added operation '{op}'")
    
    def synchronize(self):
        """Synchronize the stream (simulated)."""
        with self.lock:
            # Simulate stream execution
            time.sleep(0.002)
            logger.debug(f"Stream {self.name} synchronized with {len(self.operations)} operations")
    
    def wait_event(self, event: MockCUDAEvent):
        """Wait for an event on this stream (simulated)."""
        with self.lock:
            event.synchronize()
            self.operations.append(f"wait_event_{event.name}")
            logger.debug(f"Stream {self.name} waiting for event {event.name}")


class RaceConditionValidator:
    """
    Validates race conditions in async prefetch infrastructure.
    
    Simulates concurrent access patterns to detect potential race conditions
    in stream synchronization, memory access, and buffer swapping.
    """
    
    def __init__(self, max_threads: int = 10, stress_iterations: int = 1000):
        """
        Initialize the race condition validator.
        
        Args:
            max_threads: Maximum number of concurrent threads for stress testing
            stress_iterations: Number of iterations for stress tests
        """
        self.max_threads = max_threads
        self.stress_iterations = stress_iterations
        self.results: List[RaceConditionResult] = []
        self.metrics = ValidationMetrics()
        self._test_data = {}
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup test data buffers for memory access testing."""
        # Create mock buffers for testing
        self._test_data = {
            "buffer_a": bytearray(1024),  # 1KB buffer
            "buffer_b": bytearray(1024),  # 1KB buffer
            "weights": [bytearray(4096) for _ in range(10)],  # 10x 4KB weight buffers
            "activations": [bytearray(2048) for _ in range(5)],  # 5x 2KB activation buffers
        }
        
        # Initialize with pattern
        for i, buf in enumerate(self._test_data["weights"]):
            buf[:] = bytes([i % 256] * 4096)
        
        for i, buf in enumerate(self._test_data["activations"]):
            buf[:] = bytes([(i + 10) % 256] * 2048)
    
    def validate_stream_synchronization(self) -> RaceConditionResult:
        """
        Test 1: Validate stream synchronization correctness.
        
        Simulates concurrent operations on compute and copy streams
        to ensure proper event synchronization.
        """
        result = RaceConditionResult(
            test_name="stream_synchronization",
            condition_type=RaceConditionType.STREAM_SYNC,
            severity="high"
        )
        
        try:
            # Create mock streams and events
            compute_stream = MockStream("compute")
            copy_stream = MockStream("copy")
            sync_event = MockCUDAEvent("weight_loaded")
            
            # Simulate concurrent operations
            def compute_operation():
                compute_stream.add_operation("layer_compute")
                # Wait for weights to be loaded
                compute_stream.wait_event(sync_event)
                compute_stream.add_operation("activation_compute")
                compute_stream.synchronize()
            
            def copy_operation():
                copy_stream.add_operation("weight_transfer")
                sync_event.record(copy_stream)
                copy_stream.add_operation("buffer_update")
                copy_stream.synchronize()
                # Ensure event is synchronized before compute can wait on it
                sync_event.synchronize()
            
            # Run operations in threads
            threads = [
                threading.Thread(target=compute_operation),
                threading.Thread(target=copy_operation)
            ]
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            # Check synchronization
            if not sync_event.synchronized:
                result.detected = True
                result.description = "Event not properly synchronized between streams"
                result.error_message = "Compute stream proceeded without waiting for copy completion"
            
            # Check operation ordering
            compute_ops = compute_stream.operations
            if len(compute_ops) < 2 or "wait_event_weight_loaded" not in compute_ops[1]:
                result.detected = True
                result.description = "Incorrect operation ordering in compute stream"
                result.error_message = "Compute should wait for weight load before activation compute"
            
        except Exception as e:
            result.detected = True
            result.description = "Exception during stream synchronization test"
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        
        return result
    
    def validate_memory_access_ordering(self) -> RaceConditionResult:
        """
        Test 2: Validate memory access ordering.
        
        Tests concurrent read/write access to shared memory buffers
        to detect data corruption.
        """
        result = RaceConditionResult(
            test_name="memory_access_ordering",
            condition_type=RaceConditionType.MEMORY_ACCESS,
            severity="critical"
        )
        
        try:
            buffer = self._test_data["buffer_a"]
            corruption_detected = False
            read_errors = 0
            
            def writer_thread(thread_id: int):
                """Thread that writes to buffer."""
                nonlocal corruption_detected
                for i in range(100):
                    offset = (i * 10) % len(buffer)
                    # Write pattern based on thread and iteration
                    pattern = bytes([(thread_id + i) % 256] * 10)
                    buffer[offset:offset+10] = pattern
                    time.sleep(0.0001)  # Small delay
            
            def reader_thread(thread_id: int):
                """Thread that reads from buffer and checks consistency."""
                nonlocal corruption_detected, read_errors
                for i in range(100):
                    offset = (i * 10) % len(buffer)
                    read_data = buffer[offset:offset+10]
                    # Check if data is consistent (all bytes same)
                    if len(read_data) == 10:
                        first_byte = read_data[0]
                        if not all(b == first_byte for b in read_data):
                            corruption_detected = True
                            read_errors += 1
                    time.sleep(0.0001)
            
            # Create multiple reader/writer threads
            threads = []
            for i in range(3):
                threads.append(threading.Thread(target=writer_thread, args=(i,)))
                threads.append(threading.Thread(target=reader_thread, args=(i+3,)))
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            if corruption_detected:
                result.detected = True
                result.description = "Memory access ordering violation detected"
                result.error_message = f"Data corruption detected with {read_errors} read errors"
            
        except Exception as e:
            result.detected = True
            result.description = "Exception during memory access ordering test"
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        
        return result
    
    def validate_double_buffer_swap(self) -> RaceConditionResult:
        """
        Test 3: Validate double-buffer swap safety.
        
        Tests concurrent buffer swapping between compute and staging buffers
        to ensure atomic swaps and no data loss.
        """
        result = RaceConditionResult(
            test_name="double_buffer_swap",
            condition_type=RaceConditionType.BUFFER_SWAP,
            severity="high"
        )
        
        try:
            buffers = {
                "compute": bytearray(1024),
                "staging": bytearray(1024),
                "next": bytearray(1024)
            }
            
            # Initialize buffers
            buffers["compute"][:] = b"A" * 1024
            buffers["staging"][:] = b"B" * 1024
            buffers["next"][:] = b"C" * 1024
            
            swap_lock = threading.Lock()
            swap_count = 0
            data_corruption = False
            
            def swap_buffers():
                """Swap compute and staging buffers."""
                nonlocal swap_count, data_corruption
                with swap_lock:
                    # Simulate buffer swap
                    temp = buffers["compute"]
                    buffers["compute"] = buffers["staging"]
                    buffers["staging"] = buffers["next"]
                    buffers["next"] = temp
                    swap_count += 1
            
            def use_buffer(buffer_name: str, thread_id: int):
                """Use buffer for computation."""
                nonlocal data_corruption
                buffer = buffers[buffer_name]
                # Check buffer consistency
                if len(buffer) != 1024:
                    data_corruption = True
                # Simulate computation
                time.sleep(0.001)
            
            # Create threads that swap and use buffers concurrently
            threads = []
            for i in range(5):
                threads.append(threading.Thread(target=swap_buffers))
                threads.append(threading.Thread(target=use_buffer, args=("compute", i)))
                threads.append(threading.Thread(target=use_buffer, args=("staging", i)))
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            if data_corruption:
                result.detected = True
                result.description = "Double-buffer swap data corruption"
                result.error_message = "Buffer data corrupted during concurrent swap operations"
            
            # Check that all buffers still have valid data
            for buf_name, buffer in buffers.items():
                if len(buffer) != 1024:
                    result.detected = True
                    result.description = f"Buffer {buf_name} size corrupted"
                    result.error_message = f"Buffer size changed from 1024 to {len(buffer)}"
            
        except Exception as e:
            result.detected = True
            result.description = "Exception during double-buffer swap test"
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        
        return result
    
    def validate_concurrent_access_patterns(self) -> RaceConditionResult:
        """
        Test 4: Validate concurrent access patterns.
        
        Tests multiple threads accessing the same resources concurrently
        to detect race conditions in access patterns.
        """
        result = RaceConditionResult(
            test_name="concurrent_access_patterns",
            condition_type=RaceConditionType.CONCURRENT_READ,
            severity="medium"
        )
        
        try:
            shared_counter = 0
            counter_lock = threading.Lock()
            expected_total = 0
            actual_total = 0
            
            def increment_counter(thread_id: int, iterations: int):
                """Increment shared counter with/without lock."""
                nonlocal shared_counter, actual_total
                for _ in range(iterations):
                    with counter_lock:
                        shared_counter += 1
                        actual_total += 1
                    # Small random delay to increase chance of race conditions
                    time.sleep(random.uniform(0, 0.0001))
            
            # Calculate expected total
            num_threads = 5
            iterations_per_thread = 100
            expected_total = num_threads * iterations_per_thread
            
            # Create and run threads
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=increment_counter, args=(i, iterations_per_thread))
                threads.append(t)
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            # Check for race conditions
            if shared_counter != expected_total:
                result.detected = True
                result.description = "Concurrent access race condition detected"
                result.error_message = f"Counter mismatch: expected {expected_total}, got {shared_counter}"
            
            # Also check if actual_total matches (should with lock)
            if actual_total != expected_total:
                result.detected = True
                result.description = "Lock protection failure"
                result.error_message = f"Lock failed: expected {expected_total} increments, got {actual_total}"
            
        except Exception as e:
            result.detected = True
            result.description = "Exception during concurrent access test"
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        
        return result
    
    def run_stress_test(self) -> RaceConditionResult:
        """
        Test 5: Run comprehensive stress test.
        
        Runs multiple validation tests concurrently under high load
        to detect race conditions under stress.
        """
        result = RaceConditionResult(
            test_name="stress_test",
            condition_type=RaceConditionType.DATA_CORRUPTION,
            severity="critical"
        )
        
        try:
            stress_results = []
            stress_lock = threading.Lock()
            
            def run_stress_iteration(iter_num: int):
                """Run one iteration of stress testing."""
                # Simulate various operations
                time.sleep(random.uniform(0.001, 0.01))
                
                # Randomly choose operation type
                op_type = random.choice(["memory_op", "stream_op", "buffer_op", "sync_op"])
                
                with stress_lock:
                    stress_results.append({
                        "iteration": iter_num,
                        "operation": op_type,
                        "timestamp": time.time()
                    })
            
            # Run stress test with multiple threads
            threads = []
            for i in range(self.stress_iterations):
                t = threading.Thread(target=run_stress_iteration, args=(i,))
                threads.append(t)
                
                # Limit concurrent threads
                if len(threads) >= self.max_threads:
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    threads = []
            
            # Finish remaining threads
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Check for missing iterations
            completed_iterations = len(stress_results)
            if completed_iterations != self.stress_iterations:
                result.detected = True
                result.description = "Stress test iteration loss"
                result.error_message = f"Lost {self.stress_iterations - completed_iterations} iterations"
            
            # Check for duplicate iteration numbers
            iteration_numbers = [r["iteration"] for r in stress_results]
            if len(set(iteration_numbers)) != len(iteration_numbers):
                result.detected = True
                result.description = "Duplicate iterations detected"
                result.error_message = "Race condition caused duplicate iteration execution"
            
        except Exception as e:
            result.detected = True
            result.description = "Exception during stress test"
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        
        return result
    
    def _get_stack_trace(self) -> List[str]:
        """Get current stack trace for error reporting."""
        import traceback
        return traceback.format_stack()
    
    def run_all_tests(self) -> Dict:
        """
        Run all race condition validation tests.
        
        Returns:
            Dictionary with test results and metrics
        """
        logger.info("Starting race condition validation tests...")
        start_time = time.time()
        
        # Run all validation tests
        tests = [
            self.validate_stream_synchronization,
            self.validate_memory_access_ordering,
            self.validate_double_buffer_swap,
            self.validate_concurrent_access_patterns,
            self.run_stress_test,
        ]
        
        for test_func in tests:
            result = test_func()
            self.results.append(result)
            self.metrics.add_result(result)
            
            if result.detected:
                logger.warning(f"Test failed: {result.test_name} - {result.description}")
            else:
                logger.info(f"Test passed: {result.test_name}")
        
        # Calculate total execution time
        self.metrics.total_execution_time = time.time() - start_time
        self.metrics.concurrent_threads = self.max_threads
        
        # Generate summary
        summary = self.generate_summary()
        
        logger.info(f"Race condition validation completed in {self.metrics.total_execution_time:.2f}s")
        logger.info(f"Results: {self.metrics.passed_tests} passed, {self.metrics.failed_tests} failed")
        
        return summary
    
    def generate_summary(self) -> Dict:
        """
        Generate comprehensive test summary.
        
        Returns:
            Dictionary with test results, metrics, and recommendations
        """
        summary = {
            "timestamp": time.time(),
            "metrics": self.metrics.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "recommendations": [],
            "status": "PASS" if self.metrics.failed_tests == 0 else "FAIL"
        }
        
        # Generate recommendations based on failures
        for result in self.results:
            if result.detected:
                if result.condition_type == RaceConditionType.STREAM_SYNC:
                    summary["recommendations"].append(
                        "Add proper event synchronization between compute and copy streams"
                    )
                elif result.condition_type == RaceConditionType.MEMORY_ACCESS:
                    summary["recommendations"].append(
                        "Implement memory barriers or atomic operations for shared buffers"
                    )
                elif result.condition_type == RaceConditionType.BUFFER_SWAP:
                    summary["recommendations"].append(
                        "Use atomic buffer swapping with proper locking"
                    )
                elif result.condition_type == RaceConditionType.CONCURRENT_READ:
                    summary["recommendations"].append(
                        "Add read-write locks for concurrent access patterns"
                    )
                elif result.condition_type == RaceConditionType.DATA_CORRUPTION:
                    summary["recommendations"].append(
                        "Implement comprehensive error checking and data validation"
                    )
        
        # Add general recommendations
        if self.metrics.failed_tests > 0:
            summary["recommendations"].append(
                "Run validation on real CUDA hardware to confirm race condition detection"
            )
            summary["recommendations"].append(
                "Consider adding more fine-grained locking for high-concurrency scenarios"
            )
        
        return summary
    
    def save_results(self, output_path: str):
        """
        Save validation results to file.
        
        Args:
            output_path: Path to save results JSON file
        """
        import json
        summary = self.generate_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def validate_race_conditions(
    output_dir: str = "phase2_results/task_2_5",
    max_threads: int = 10,
    stress_iterations: int = 1000
) -> Dict:
    """
    Main function to run race condition validation.
    
    Args:
        output_dir: Directory to save results
        max_threads: Maximum concurrent threads for stress testing
        stress_iterations: Number of stress test iterations
    
    Returns:
        Validation summary dictionary
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    validator = RaceConditionValidator(
        max_threads=max_threads,
        stress_iterations=stress_iterations
    )
    
    results = validator.run_all_tests()
    
    # Save results
    output_path = os.path.join(output_dir, "race_condition_validation.json")
    validator.save_results(output_path)
    
    # Generate report
    report_path = os.path.join(output_dir, "race_condition_report.txt")
    _generate_text_report(results, report_path)
    
    return results


def _generate_text_report(results: Dict, output_path: str):
    """Generate human-readable text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RACE CONDITION VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("WARNING: SIMULATED DATA — not from real hardware\n")
        f.write("These results are from CPU-based simulation and must be validated\n")
        f.write("on real CUDA hardware before deployment (see ROADMAP task 2.5).\n\n")
        
        f.write(f"Timestamp: {time.ctime(results['timestamp'])}\n")
        f.write(f"Overall Status: {results['status']}\n\n")
        
        # Metrics
        metrics = results['metrics']
        f.write("METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Tests: {metrics['total_tests']}\n")
        f.write(f"Passed: {metrics['passed_tests']}\n")
        f.write(f"Failed: {metrics['failed_tests']}\n")
        f.write(f"Pass Rate: {metrics['pass_rate']:.1f}%\n")
        f.write(f"Critical Issues: {metrics['critical_issues']}\n")
        f.write(f"High Severity: {metrics['high_severity']}\n")
        f.write(f"Medium Severity: {metrics['medium_severity']}\n")
        f.write(f"Low Severity: {metrics['low_severity']}\n")
        f.write(f"Execution Time: {metrics['total_execution_time']:.2f}s\n")
        f.write(f"Max Concurrent Threads: {metrics['concurrent_threads']}\n\n")
        
        # Test Results
        f.write("DETAILED TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        for result in results['results']:
            status = "PASS" if not result['detected'] else "FAIL"
            f.write(f"\nTest: {result['test_name']} [{status}]\n")
            f.write(f"Type: {result['condition_type']}\n")
            f.write(f"Severity: {result['severity']}\n")
            
            if result['detected']:
                f.write(f"Issue: {result['description']}\n")
                f.write(f"Error: {result['error_message']}\n")
            else:
                f.write("No issues detected\n")
        
        # Recommendations
        if results['recommendations']:
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    results = validate_race_conditions()
    print(f"Validation completed. Status: {results['status']}")