"""
Output Validation for Async Prefetch Infrastructure

WARNING: SIMULATED DATA — not from real hardware
This validator compares outputs from async prefetch execution vs baseline
(eager) execution. All comparisons are based on simulated data and must be
validated on real CUDA hardware before deployment (see ROADMAP task 2.6).

Validates:
1. Output correctness of async prefetch vs eager baseline
2. Numerical equivalence within tolerance
3. Integration with dual stream infrastructure
4. Race condition impact on output correctness
5. Stress test output consistency
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cuda_graph.output_validation import (
    OutputValidator as BaseOutputValidator,
    ValidationResult,
    ValidationMode
)


class AsyncValidationMode(Enum):
    """Validation modes specific to async prefetch."""
    ASYNC_VS_EAGER = "async_vs_eager"  # Compare async prefetch vs eager baseline
    STREAM_SYNC = "stream_sync"  # Validate stream synchronization correctness
    MEMORY_CONSISTENCY = "memory_consistency"  # Check memory access consistency
    STRESS_TEST = "stress_test"  # Validate under stress conditions
    RACE_CONDITION_IMPACT = "race_condition_impact"  # Check race condition effects


@dataclass
class AsyncValidationResult:
    """Validation result for async prefetch."""
    
    passed: bool
    validation_mode: AsyncValidationMode
    comparison_count: int
    passed_count: int
    failed_count: int
    max_absolute_error: float
    max_relative_error: float
    stream_sync_correct: bool = True
    memory_consistent: bool = True
    race_conditions_detected: int = 0
    stress_test_passed: bool = True
    async_overhead_ms: float = 0.0
    memory_transfer_overlap_percent: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "validation_mode": self.validation_mode.value,
            "comparison_count": self.comparison_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "stream_sync_correct": self.stream_sync_correct,
            "memory_consistent": self.memory_consistent,
            "race_conditions_detected": self.race_conditions_detected,
            "stress_test_passed": self.stress_test_passed,
            "async_overhead_ms": self.async_overhead_ms,
            "memory_transfer_overlap_percent": self.memory_transfer_overlap_percent,
            "details": self.details
        }


class MockAsyncPrefetchEngine:
    """
    Mock async prefetch engine for validation testing.
    
    Simulates async weight prefetch with dual streams and pinned memory
    for output comparison testing.
    """
    
    def __init__(self, use_async: bool = True, enable_race_conditions: bool = False):
        """
        Initialize mock async prefetch engine.
        
        Args:
            use_async: Whether to use async prefetch (vs eager baseline)
            enable_race_conditions: Whether to simulate race conditions
        """
        self.use_async = use_async
        self.enable_race_conditions = enable_race_conditions
        self.stream_sync_correct = True
        self.memory_consistent = True
        self.race_conditions = 0
        
        # Mock streams
        self.compute_stream = "compute_stream"
        self.copy_stream = "copy_stream"
        
        # Mock buffers
        self.weight_buffers = {}
        self.activation_buffers = {}
        
        # Statistics
        self.execution_time_ms = 0.0
        self.overlap_percent = 0.0
        
    def initialize_buffers(self, layer_count: int, buffer_size: int = 4096):
        """Initialize mock buffers for testing."""
        for i in range(layer_count):
            self.weight_buffers[f"layer_{i}"] = {
                "compute": torch.randn(buffer_size // 4, dtype=torch.float32),
                "staging": torch.randn(buffer_size // 4, dtype=torch.float32),
                "next": torch.randn(buffer_size // 4, dtype=torch.float32),
            }
            self.activation_buffers[f"layer_{i}"] = {
                "input": torch.randn(buffer_size // 8, dtype=torch.float32),
                "output": torch.randn(buffer_size // 8, dtype=torch.float32),
            }
    
    def execute_layer(self, layer_idx: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute a single layer with optional async prefetch.
        
        Args:
            layer_idx: Layer index to execute
            input_tensor: Input tensor for the layer
            
        Returns:
            Output tensor from layer execution
        """
        start_time = time.time()
        
        if self.use_async:
            # Simulate async execution with dual streams
            output = self._execute_async(layer_idx, input_tensor)
        else:
            # Simulate eager (baseline) execution
            output = self._execute_eager(layer_idx, input_tensor)
        
        # Add some execution time
        time.sleep(0.001)  # Simulate 1ms computation
        self.execution_time_ms = (time.time() - start_time) * 1000
        
        # Simulate overlap if using async
        if self.use_async:
            self.overlap_percent = 0.3  # 30% overlap (simulated)
        
        return output
    
    def _execute_eager(self, layer_idx: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute layer in eager mode (baseline)."""
        # Simple linear transformation for testing
        weight_key = f"layer_{layer_idx}"
        if weight_key in self.weight_buffers:
            weight = self.weight_buffers[weight_key]["compute"]
            # Ensure compatible dimensions
            if len(weight.shape) == 1 and len(input_tensor.shape) == 1:
                # Simple dot product simulation
                output = torch.dot(input_tensor, weight[:input_tensor.shape[0]])
                return output.unsqueeze(0) if output.dim() == 0 else output
        return input_tensor * 1.5  # Fallback transformation
    
    def _execute_async(self, layer_idx: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute layer with async prefetch simulation."""
        # Simulate dual stream execution
        compute_start = time.time()
        
        # Get current weight (already loaded)
        weight_key = f"layer_{layer_idx}"
        current_weight = None
        if weight_key in self.weight_buffers:
            current_weight = self.weight_buffers[weight_key]["compute"]
        
        # Simulate computation on compute stream
        if current_weight is not None:
            if len(current_weight.shape) == 1 and len(input_tensor.shape) == 1:
                output = torch.dot(input_tensor, current_weight[:input_tensor.shape[0]])
            else:
                output = input_tensor * 1.5
        else:
            output = input_tensor * 1.5
        
        # Simulate async weight prefetch for next layer
        if layer_idx + 1 in [int(k.split('_')[1]) for k in self.weight_buffers.keys()]:
            next_weight_key = f"layer_{layer_idx + 1}"
            if next_weight_key in self.weight_buffers:
                # Simulate weight transfer on copy stream
                time.sleep(0.0003)  # Simulate transfer time
                
                # Swap buffers for next layer
                self._swap_buffers(next_weight_key)
        
        # Simulate race conditions if enabled
        if self.enable_race_conditions:
            self._simulate_race_conditions(layer_idx)
        
        compute_time = (time.time() - compute_start) * 1000
        
        # Update statistics
        self.execution_time_ms = compute_time
        
        return output.unsqueeze(0) if output.dim() == 0 else output
    
    def _swap_buffers(self, layer_key: str):
        """Simulate buffer swapping for async prefetch."""
        if layer_key in self.weight_buffers:
            buffers = self.weight_buffers[layer_key]
            # Rotate buffers: staging -> compute, next -> staging, compute -> next
            temp = buffers["compute"]
            buffers["compute"] = buffers["staging"]
            buffers["staging"] = buffers["next"]
            buffers["next"] = temp
            
            # Check for consistency
            if not torch.allclose(buffers["compute"], buffers["staging"], rtol=1e-5):
                self.memory_consistent = False
    
    def _simulate_race_conditions(self, layer_idx: int):
        """Simulate race conditions for testing."""
        import random
        
        # Randomly introduce race conditions
        if random.random() < 0.1:  # 10% chance
            self.race_conditions += 1
            
            # Simulate different types of race conditions
            race_type = random.choice(["stream_sync", "memory_access", "buffer_swap"])
            
            if race_type == "stream_sync":
                self.stream_sync_correct = False
            elif race_type == "memory_access":
                self.memory_consistent = False
            elif race_type == "buffer_swap":
                # Corrupt a buffer
                if f"layer_{layer_idx}" in self.weight_buffers:
                    buffer = self.weight_buffers[f"layer_{layer_idx}"]["compute"]
                    if buffer.numel() > 0:
                        buffer[0] = torch.tensor(float('nan'))
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "use_async": self.use_async,
            "execution_time_ms": self.execution_time_ms,
            "overlap_percent": self.overlap_percent,
            "stream_sync_correct": self.stream_sync_correct,
            "memory_consistent": self.memory_consistent,
            "race_conditions": self.race_conditions,
            "buffer_count": len(self.weight_buffers),
        }


class AsyncOutputValidator(BaseOutputValidator):
    """
    Output validator for async prefetch infrastructure.
    
    Extends the base output validator with async-specific validation
    and integrates with the async prefetch engine.
    """
    
    def __init__(
        self,
        absolute_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-4,
        enable_race_condition_testing: bool = False,
        stress_test_iterations: int = 100
    ):
        """
        Initialize async output validator.
        
        Args:
            absolute_tolerance: Absolute tolerance for numerical comparison
            relative_tolerance: Relative tolerance for numerical comparison
            enable_race_condition_testing: Whether to test with race conditions
            stress_test_iterations: Number of iterations for stress testing
        """
        super().__init__(absolute_tolerance, relative_tolerance)
        self.enable_race_condition_testing = enable_race_condition_testing
        self.stress_test_iterations = stress_test_iterations
        self.async_engine = None
        self.eager_engine = None
        
    def initialize_engines(self, layer_count: int = 8, buffer_size: int = 4096):
        """Initialize async and eager engines for comparison."""
        # Create async engine (with optional race conditions)
        self.async_engine = MockAsyncPrefetchEngine(
            use_async=True,
            enable_race_conditions=self.enable_race_condition_testing
        )
        self.async_engine.initialize_buffers(layer_count, buffer_size)
        
        # Create eager engine (baseline)
        self.eager_engine = MockAsyncPrefetchEngine(
            use_async=False,
            enable_race_conditions=False
        )
        self.eager_engine.initialize_buffers(layer_count, buffer_size)
        
        # Ensure both engines have the same initial weights
        for layer_key in self.async_engine.weight_buffers:
            if layer_key in self.eager_engine.weight_buffers:
                # Copy weights from async to eager to ensure same initialization
                for buffer_type in ["compute", "staging", "next"]:
                    self.eager_engine.weight_buffers[layer_key][buffer_type] = \
                        self.async_engine.weight_buffers[layer_key][buffer_type].clone()
    
    def validate_async_vs_eager(
        self,
        input_tensor: torch.Tensor,
        layer_indices: List[int] = None,
        num_iterations: int = 10
    ) -> AsyncValidationResult:
        """
        Validate async prefetch outputs vs eager baseline.
        
        Args:
            input_tensor: Input tensor for validation
            layer_indices: List of layer indices to validate (None for all)
            num_iterations: Number of iterations to run
            
        Returns:
            AsyncValidationResult with comparison results
        """
        if self.async_engine is None or self.eager_engine is None:
            self.initialize_engines()
        
        if layer_indices is None:
            layer_indices = list(range(len(self.async_engine.weight_buffers)))
        
        # Run multiple iterations
        all_async_outputs = []
        all_eager_outputs = []
        execution_stats = []
        
        for iteration in range(num_iterations):
            async_outputs = []
            eager_outputs = []
            
            for layer_idx in layer_indices:
                # Execute with async prefetch
                async_output = self.async_engine.execute_layer(layer_idx, input_tensor)
                async_outputs.append(async_output)
                
                # Execute with eager baseline
                eager_output = self.eager_engine.execute_layer(layer_idx, input_tensor)
                eager_outputs.append(eager_output)
            
            all_async_outputs.append(async_outputs)
            all_eager_outputs.append(eager_outputs)
            
            # Collect execution stats
            execution_stats.append({
                "async": self.async_engine.get_execution_stats(),
                "eager": self.eager_engine.get_execution_stats(),
            })
        
        # Compare outputs
        comparison_results = []
        for iter_idx in range(num_iterations):
            for layer_idx in range(len(layer_indices)):
                async_out = all_async_outputs[iter_idx][layer_idx]
                eager_out = all_eager_outputs[iter_idx][layer_idx]
                
                comparison = self._compare_tensors(
                    async_out, eager_out, f"layer_{layer_indices[layer_idx]}_iter_{iter_idx}"
                )
                comparison_results.append(comparison)
        
        # Check if all comparisons passed
        all_passed = all(r["passed"] for r in comparison_results)
        
        # Get async engine statistics
        async_stats = self.async_engine.get_execution_stats()
        eager_stats = self.eager_engine.get_execution_stats()
        
        # Calculate overhead
        async_time = async_stats.get("execution_time_ms", 0)
        eager_time = eager_stats.get("execution_time_ms", 0)
        overhead = max(0, async_time - eager_time)
        
        # Create validation result
        max_abs_error = max(r["max_absolute_error"] for r in comparison_results) if comparison_results else 0
        max_rel_error = max(r["max_relative_error"] for r in comparison_results) if comparison_results else 0
        
        result = AsyncValidationResult(
            passed=all_passed,
            validation_mode=AsyncValidationMode.ASYNC_VS_EAGER,
            comparison_count=len(comparison_results),
            passed_count=sum(1 for r in comparison_results if r["passed"]),
            failed_count=sum(1 for r in comparison_results if not r["passed"]),
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            stream_sync_correct=async_stats.get("stream_sync_correct", True),
            memory_consistent=async_stats.get("memory_consistent", True),
            race_conditions_detected=async_stats.get("race_conditions", 0),
            stress_test_passed=True,  # Will be set by stress test
            async_overhead_ms=overhead,
            memory_transfer_overlap_percent=async_stats.get("overlap_percent", 0),
            details={
                "comparison_results": comparison_results,
                "execution_stats": execution_stats,
                "layer_indices": layer_indices,
                "num_iterations": num_iterations,
                "input_shape": list(input_tensor.shape),
                "input_dtype": str(input_tensor.dtype),
            }
        )
        
        return result
    
    def validate_stream_synchronization(
        self,
        input_tensor: torch.Tensor,
        num_layers: int = 4,
        iterations: int = 5
    ) -> AsyncValidationResult:
        """
        Validate stream synchronization correctness.
        
        Args:
            input_tensor: Input tensor for validation
            num_layers: Number of layers to test
            iterations: Number of iterations per layer
            
        Returns:
            AsyncValidationResult with stream sync validation
        """
        if self.async_engine is None:
            self.initialize_engines(num_layers)
        
        sync_issues = 0
        execution_times = []
        
        for layer_idx in range(num_layers):
            for iter_idx in range(iterations):
                start_time = time.time()
                output = self.async_engine.execute_layer(layer_idx, input_tensor)
                execution_time = (time.time() - start_time) * 1000
                execution_times.append(execution_time)
                
                # Check stream sync
                stats = self.async_engine.get_execution_stats()
                if not stats.get("stream_sync_correct", True):
                    sync_issues += 1
        
        # Create validation result
        result = AsyncValidationResult(
            passed=sync_issues == 0,
            validation_mode=AsyncValidationMode.STREAM_SYNC,
            stream_sync_correct=sync_issues == 0,
            memory_consistent=True,
            race_conditions_detected=0,
            stress_test_passed=True,
            async_overhead_ms=np.mean(execution_times) if execution_times else 0,
            memory_transfer_overlap_percent=0,
            comparison_count=num_layers * iterations,
            passed_count=(num_layers * iterations) - sync_issues,
            failed_count=sync_issues,
            max_absolute_error=0,
            max_relative_error=0,
            details={
                "sync_issues": sync_issues,
                "execution_times_ms": execution_times,
                "num_layers": num_layers,
                "iterations": iterations,
                "avg_execution_time": np.mean(execution_times) if execution_times else 0,
            }
        )
        
        return result
    
    def validate_memory_consistency(
        self,
        input_tensor: torch.Tensor,
        buffer_size: int = 1024,
        iterations: int = 20
    ) -> AsyncValidationResult:
        """
        Validate memory consistency under concurrent access.
        
        Args:
            input_tensor: Input tensor for validation
            buffer_size: Size of memory buffers to test
            iterations: Number of iterations
            
        Returns:
            AsyncValidationResult with memory consistency validation
        """
        # Initialize with more layers for memory testing
        self.initialize_engines(layer_count=8, buffer_size=buffer_size)
        
        consistency_issues = 0
        buffer_checks = []
        
        for iteration in range(iterations):
            # Execute all layers
            for layer_idx in range(8):
                output = self.async_engine.execute_layer(layer_idx, input_tensor)
                
                # Check buffer consistency
                layer_key = f"layer_{layer_idx}"
                if layer_key in self.async_engine.weight_buffers:
                    buffers = self.async_engine.weight_buffers[layer_key]
                    
                    # Check that compute buffer is valid (not NaN or Inf)
                    compute_buffer = buffers["compute"]
                    if torch.isnan(compute_buffer).any() or torch.isinf(compute_buffer).any():
                        consistency_issues += 1
                        buffer_checks.append({
                            "iteration": iteration,
                            "layer": layer_idx,
                            "issue": "invalid_buffer_value"
                        })
                    
                    # Check that buffers are different (should be after swaps)
                    if iteration > 0:
                        if torch.allclose(buffers["compute"], buffers["staging"], rtol=1e-5):
                            consistency_issues += 1
                            buffer_checks.append({
                                "iteration": iteration,
                                "layer": layer_idx,
                                "issue": "identical_buffers"
                            })
        
        # Get final memory consistency from engine
        stats = self.async_engine.get_execution_stats()
        memory_consistent = stats.get("memory_consistent", True) and consistency_issues == 0
        
        result = AsyncValidationResult(
            passed=memory_consistent,
            validation_mode=AsyncValidationMode.MEMORY_CONSISTENCY,
            stream_sync_correct=True,
            memory_consistent=memory_consistent,
            race_conditions_detected=0,
            stress_test_passed=True,
            async_overhead_ms=0,
            memory_transfer_overlap_percent=0,
            comparison_count=iterations * 8,
            passed_count=(iterations * 8) - consistency_issues,
            failed_count=consistency_issues,
            max_absolute_error=0,
            max_relative_error=0,
            details={
                "consistency_issues": consistency_issues,
                "buffer_checks": buffer_checks,
                "buffer_size": buffer_size,
                "iterations": iterations,
                "memory_consistent": memory_consistent,
            }
        )
        
        return result
    
    def run_stress_test(
        self,
        input_tensor: torch.Tensor,
        num_layers: int = 16,
        iterations: int = None
    ) -> AsyncValidationResult:
        """
        Run stress test on async prefetch infrastructure.
        
        Args:
            input_tensor: Input tensor for stress testing
            num_layers: Number of layers to stress test
            iterations: Number of iterations (defaults to stress_test_iterations)
            
        Returns:
            AsyncValidationResult with stress test results
        """
        if iterations is None:
            iterations = self.stress_test_iterations
        
        # Re-initialize with stress test configuration
        self.initialize_engines(layer_count=num_layers)
        
        failures = 0
        execution_stats = []
        outputs = []
        
        for iteration in range(iterations):
            iteration_outputs = []
            
            for layer_idx in range(num_layers):
                try:
                    output = self.async_engine.execute_layer(layer_idx, input_tensor)
                    iteration_outputs.append(output)
                except Exception as e:
                    failures += 1
                    iteration_outputs.append(None)
            
            outputs.append(iteration_outputs)
            
            # Collect stats
            stats = self.async_engine.get_execution_stats()
            execution_stats.append(stats)
        
        # Check for consistency across iterations
        consistency_issues = 0
        if len(outputs) > 1:
            # Compare first and last iteration outputs
            for layer_idx in range(num_layers):
                if outputs[0][layer_idx] is not None and outputs[-1][layer_idx] is not None:
                    if not torch.allclose(outputs[0][layer_idx], outputs[-1][layer_idx], rtol=1e-4):
                        consistency_issues += 1
        
        stress_test_passed = failures == 0 and consistency_issues == 0
        
        result = AsyncValidationResult(
            passed=stress_test_passed,
            validation_mode=AsyncValidationMode.STRESS_TEST,
            stream_sync_correct=True,
            memory_consistent=True,
            race_conditions_detected=0,
            stress_test_passed=stress_test_passed,
            async_overhead_ms=0,
            memory_transfer_overlap_percent=0,
            comparison_count=iterations * num_layers,
            passed_count=(iterations * num_layers) - failures - consistency_issues,
            failed_count=failures + consistency_issues,
            max_absolute_error=0,
            max_relative_error=0,
            details={
                "failures": failures,
                "consistency_issues": consistency_issues,
                "num_layers": num_layers,
                "iterations": iterations,
                "execution_stats": execution_stats,
                "stress_test_passed": stress_test_passed,
            }
        )
        
        return result
    
    def validate_all_modes(
        self,
        input_tensor: torch.Tensor,
        output_dir: str = "phase2_results/task_2_6"
    ) -> Dict[str, AsyncValidationResult]:
        """
        Run all validation modes and save results.
        
        Args:
            input_tensor: Input tensor for validation
            output_dir: Directory to save results
            
        Returns:
            Dictionary of validation results by mode
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Run all validation modes
        validation_modes = [
            ("async_vs_eager", lambda: self.validate_async_vs_eager(input_tensor)),
            ("stream_sync", lambda: self.validate_stream_synchronization(input_tensor)),
            ("memory_consistency", lambda: self.validate_memory_consistency(input_tensor)),
            ("stress_test", lambda: self.run_stress_test(input_tensor)),
        ]
        
        for mode_name, validation_func in validation_modes:
            print(f"Running {mode_name} validation...")
            result = validation_func()
            results[mode_name] = result
            
            # Save individual result
            result_path = os.path.join(output_dir, f"{mode_name}_validation.json")
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Generate comprehensive report
        self._generate_async_validation_report(results, output_dir)
        
        return results
    
    def _generate_async_validation_report(
        self,
        results: Dict[str, AsyncValidationResult],
        output_dir: str
    ):
        """Generate comprehensive validation report."""
        import os
        report_path = os.path.join(output_dir, "async_validation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ASYNC PREFETCH OUTPUT VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("WARNING: SIMULATED DATA — not from real hardware\n")
            f.write("These results are from CPU-based simulation and must be validated\n")
            f.write("on real CUDA hardware before deployment (see ROADMAP task 2.6).\n\n")
            
            f.write(f"Generated: {time.ctime()}\n")
            f.write(f"Validation Modes: {len(results)}\n\n")
            
            # Overall summary
            total_tests = sum(r.comparison_count for r in results.values())
            passed_tests = sum(r.passed_count for r in results.values())
            failed_tests = sum(r.failed_count for r in results.values())
            
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {failed_tests}\n")
            f.write(f"Pass Rate: {passed_tests/total_tests*100:.1f}%\n\n")
            
            # Detailed results by mode
            f.write("DETAILED RESULTS BY VALIDATION MODE\n")
            f.write("-" * 40 + "\n")
            
            for mode_name, result in results.items():
                f.write(f"\n{mode_name.upper()}:\n")
                f.write(f"  Status: {'PASS' if result.passed else 'FAIL'}\n")
                f.write(f"  Tests: {result.passed_count} passed, {result.failed_count} failed\n")
                f.write(f"  Mode: {result.validation_mode.value}\n")
                
                if hasattr(result, 'stream_sync_correct'):
                    f.write(f"  Stream Sync: {'OK' if result.stream_sync_correct else 'ISSUES'}\n")
                
                if hasattr(result, 'memory_consistent'):
                    f.write(f"  Memory Consistency: {'OK' if result.memory_consistent else 'ISSUES'}\n")
                
                if hasattr(result, 'race_conditions_detected'):
                    f.write(f"  Race Conditions: {result.race_conditions_detected}\n")
                
                if hasattr(result, 'async_overhead_ms') and result.async_overhead_ms > 0:
                    f.write(f"  Async Overhead: {result.async_overhead_ms:.2f} ms\n")
                
                if hasattr(result, 'memory_transfer_overlap_percent'):
                    f.write(f"  Memory Overlap: {result.memory_transfer_overlap_percent:.1f}%\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            issues_found = False
            for mode_name, result in results.items():
                if not result.passed:
                    issues_found = True
                    f.write(f"\n{mode_name.upper()} issues:\n")
                    f.write(f"  - {result.failed_count} tests failed\n")
                    
                    if hasattr(result, 'stream_sync_correct') and not result.stream_sync_correct:
                        f.write("  - Stream synchronization issues detected\n")
                    
                    if hasattr(result, 'memory_consistent') and not result.memory_consistent:
                        f.write("  - Memory consistency issues detected\n")
            
            if not issues_found:
                f.write("All validation tests passed in simulation.\n")
                f.write("Note: Real CUDA hardware validation still required.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


def create_async_validator(
    enable_race_condition_testing: bool = False,
    stress_test_iterations: int = 100
) -> AsyncOutputValidator:
    """
    Create a configured async output validator.
    
    Args:
        enable_race_condition_testing: Whether to enable race condition testing
        stress_test_iterations: Number of stress test iterations
    
    Returns:
        Configured AsyncOutputValidator instance
    """
    return AsyncOutputValidator(
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        enable_race_condition_testing=enable_race_condition_testing,
        stress_test_iterations=stress_test_iterations
    )


if __name__ == "__main__":
    # Example usage
    validator = create_async_validator()
    
    # Create test input
    input_tensor = torch.randn(128, dtype=torch.float32)
    
    # Run all validations
    results = validator.validate_all_modes(
        input_tensor=input_tensor,
        output_dir="phase2_results/task_2_6"
    )
    
    print(f"Validation completed. Results saved to phase2_results/task_2_6/")
    
    # Print summary
    total_passed = sum(1 for r in results.values() if r.passed)
    print(f"\nSummary: {total_passed}/{len(results)} validation modes passed")