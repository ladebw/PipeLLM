"""
Output Validation for CUDA Graph Compilation (Phase 1, Task 1.5)

This module validates that CUDA graph execution produces identical outputs
to eager (baseline) execution for LLM decode loops.

Key features:
1. Numerical equivalence validation with configurable tolerance
2. Support for multiple context length buckets (512, 1024, 2048, 4096)
3. Statistical analysis of output differences
4. Automated validation across different model configurations
5. Detailed reporting of validation results
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation modes for different levels of rigor."""
    QUICK = "quick"  # Single forward pass comparison
    THOROUGH = "thorough"  # Multiple forward passes with different inputs
    STRESS = "stress"  # Many iterations with random inputs


@dataclass
class ValidationResult:
    """Results from output validation."""
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    mean_relative_error: float
    num_tensors_compared: int
    num_elements_compared: int
    validation_time_ms: float
    sequence_length: int
    batch_size: int
    graph_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_relative_error": self.mean_relative_error,
            "num_tensors_compared": self.num_tensors_compared,
            "num_elements_compared": self.num_elements_compared,
            "validation_time_ms": self.validation_time_ms,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "graph_type": self.graph_type,
            "warnings": self.warnings
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class OutputValidator:
    """
    Validates that CUDA graph execution produces identical outputs to eager execution.
    
    This class provides comprehensive validation for CUDA graph compilation,
    ensuring numerical equivalence across different context lengths and model
    configurations.
    """
    
    def __init__(
        self,
        absolute_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-4,
        enable_nan_check: bool = True,
        enable_inf_check: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the output validator.
        
        Args:
            absolute_tolerance: Maximum allowed absolute difference
            relative_tolerance: Maximum allowed relative difference
            enable_nan_check: Check for NaN values in outputs
            enable_inf_check: Check for infinite values in outputs
            device: Device to run validation on
        """
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.enable_nan_check = enable_nan_check
        self.enable_inf_check = enable_inf_check
        self.device = device
        
        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
    
    def _compare_tensors(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_name: str = "tensor"
    ) -> Dict[str, Any]:
        """
        Compare two tensors and compute error metrics.
        
        Returns:
            Dictionary with comparison results
        """
        if tensor_a.shape != tensor_b.shape:
            return {
                "error": f"Shape mismatch: {tensor_a.shape} != {tensor_b.shape}",
                "passed": False
            }
        
        if tensor_a.dtype != tensor_b.dtype:
            return {
                "error": f"Dtype mismatch: {tensor_a.dtype} != {tensor_b.dtype}",
                "passed": False
            }
        
        # Move to CPU for comparison if needed
        if tensor_a.device != tensor_b.device:
            tensor_a = tensor_a.cpu()
            tensor_b = tensor_b.cpu()
        
        # Flatten for element-wise comparison
        a_flat = tensor_a.flatten().float()
        b_flat = tensor_b.flatten().float()
        
        # Check for NaN
        if self.enable_nan_check:
            a_nan = torch.isnan(a_flat).any()
            b_nan = torch.isnan(b_flat).any()
            if a_nan or b_nan:
                return {
                    "error": f"NaN values detected in {tensor_name}",
                    "passed": False,
                    "a_has_nan": a_nan.item(),
                    "b_has_nan": b_nan.item()
                }
        
        # Check for Inf
        if self.enable_inf_check:
            a_inf = torch.isinf(a_flat).any()
            b_inf = torch.isinf(b_flat).any()
            if a_inf or b_inf:
                return {
                    "error": f"Infinite values detected in {tensor_name}",
                    "passed": False,
                    "a_has_inf": a_inf.item(),
                    "b_has_inf": b_inf.item()
                }
        
        # Compute errors
        absolute_error = torch.abs(a_flat - b_flat)
        relative_error = torch.abs(a_flat - b_flat) / (torch.abs(b_flat) + 1e-8)
        
        # Find max errors
        max_abs_error = absolute_error.max().item()
        max_rel_error = relative_error.max().item()
        
        # Find mean errors
        mean_abs_error = absolute_error.mean().item()
        mean_rel_error = relative_error.mean().item()
        
        # Check if errors are within tolerance
        abs_passed = max_abs_error <= self.absolute_tolerance
        rel_passed = max_rel_error <= self.relative_tolerance
        
        passed = abs_passed and rel_passed
        
        return {
            "passed": passed,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
            "mean_absolute_error": mean_abs_error,
            "mean_relative_error": mean_rel_error,
            "num_elements": a_flat.numel(),
            "abs_passed": abs_passed,
            "rel_passed": rel_passed,
            "tensor_name": tensor_name,
            "shape": tensor_a.shape,
            "dtype": str(tensor_a.dtype)
        }
    
    def _compare_outputs(
        self,
        outputs_a,
        outputs_b,
        output_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two output structures (can be tensors, tuples, lists, or dicts).
        
        Returns:
            Dictionary with aggregated comparison results
        """
        results = []
        all_passed = True
        total_elements = 0
        
        def recursive_compare(a, b, path=""):
            nonlocal all_passed, total_elements
            
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                tensor_name = path if path else "output"
                if output_names and len(results) < len(output_names):
                    tensor_name = output_names[len(results)]
                
                result = self._compare_tensors(a, b, tensor_name)
                results.append(result)
                
                if not result.get("passed", False):
                    all_passed = False
                
                total_elements += result.get("num_elements", 0)
                
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                if len(a) != len(b):
                    results.append({
                        "error": f"Length mismatch at {path}: {len(a)} != {len(b)}",
                        "passed": False
                    })
                    all_passed = False
                    return
                
                for i, (item_a, item_b) in enumerate(zip(a, b)):
                    recursive_compare(item_a, item_b, f"{path}[{i}]")
                    
            elif isinstance(a, dict) and isinstance(b, dict):
                if set(a.keys()) != set(b.keys()):
                    missing_keys = set(a.keys()) - set(b.keys())
                    extra_keys = set(b.keys()) - set(a.keys())
                    results.append({
                        "error": f"Key mismatch at {path}: missing={missing_keys}, extra={extra_keys}",
                        "passed": False
                    })
                    all_passed = False
                    return
                
                for key in a.keys():
                    recursive_compare(a[key], b[key], f"{path}.{key}")
                    
            else:
                # For non-tensor values, check equality
                if a != b:
                    results.append({
                        "error": f"Value mismatch at {path}: {a} != {b}",
                        "passed": False
                    })
                    all_passed = False
        
        recursive_compare(outputs_a, outputs_b)
        
        # Aggregate results
        if results and all(isinstance(r, dict) and "max_absolute_error" in r for r in results):
            max_abs_error = max(r["max_absolute_error"] for r in results if "max_absolute_error" in r)
            max_rel_error = max(r["max_relative_error"] for r in results if "max_relative_error" in r)
            mean_abs_error = np.mean([r["mean_absolute_error"] for r in results if "mean_absolute_error" in r])
            mean_rel_error = np.mean([r["mean_relative_error"] for r in results if "mean_relative_error" in r])
        else:
            max_abs_error = max_rel_error = mean_abs_error = mean_rel_error = 0.0
        
        return {
            "all_passed": all_passed,
            "results": results,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
            "mean_absolute_error": mean_abs_error,
            "mean_relative_error": mean_rel_error,
            "num_tensors_compared": len([r for r in results if isinstance(r, dict) and "tensor_name" in r]),
            "num_elements_compared": total_elements
        }
    
    def validate_single_forward(
        self,
        eager_func: Callable,
        graph_func: Callable,
        inputs: List[torch.Tensor],
        sequence_length: int,
        batch_size: int = 1,
        graph_type: Optional[str] = None,
        output_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a single forward pass.
        
        Args:
            eager_func: Function that performs eager execution
            graph_func: Function that performs graph execution
            inputs: List of input tensors
            sequence_length: Sequence length for validation
            batch_size: Batch size for validation
            graph_type: Type of graph used (for reporting)
            output_names: Optional names for output tensors
            
        Returns:
            ValidationResult with comparison results
        """
        start_time = time.time()
        
        # Run eager execution
        eager_outputs = eager_func(*inputs)
        
        # Run graph execution
        graph_outputs = graph_func(*inputs)
        
        # Compare outputs
        comparison = self._compare_outputs(eager_outputs, graph_outputs, output_names)
        
        validation_time_ms = (time.time() - start_time) * 1000
        
        # Check if passed
        passed = comparison["all_passed"]
        
        # Collect warnings
        warnings = []
        if comparison["max_absolute_error"] > self.absolute_tolerance * 0.1:
            warnings.append(f"High absolute error: {comparison['max_absolute_error']:.2e}")
        if comparison["max_relative_error"] > self.relative_tolerance * 0.1:
            warnings.append(f"High relative error: {comparison['max_relative_error']:.2e}")
        
        return ValidationResult(
            passed=passed,
            max_absolute_error=comparison["max_absolute_error"],
            max_relative_error=comparison["max_relative_error"],
            mean_absolute_error=comparison["mean_absolute_error"],
            mean_relative_error=comparison["mean_relative_error"],
            num_tensors_compared=comparison["num_tensors_compared"],
            num_elements_compared=comparison["num_elements_compared"],
            validation_time_ms=validation_time_ms,
            sequence_length=sequence_length,
            batch_size=batch_size,
            graph_type=graph_type,
            error_details=comparison["results"] if not passed else None,
            warnings=warnings
        )
    
    def validate_multiple_context_lengths(
        self,
        eager_func: Callable,
        graph_func: Callable,
        context_lengths: List[int] = None,
        batch_size: int = 1,
        hidden_size: int = 4096,
        num_iterations: int = 3,
        graph_type_mapper: Optional[Callable[[int], str]] = None
    ) -> Dict[int, ValidationResult]:
        """
        Validate across multiple context lengths.
        
        Args:
            eager_func: Function that performs eager execution
            graph_func: Function that performs graph execution
            context_lengths: List of sequence lengths to test
            batch_size: Batch size for validation
            hidden_size: Hidden size for input tensors
            num_iterations: Number of iterations per context length
            graph_type_mapper: Function that maps seq_len to graph type
            
        Returns:
            Dictionary mapping context lengths to validation results
        """
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096]
        
        results = {}
        
        for seq_len in context_lengths:
            logger.info(f"Validating context length: {seq_len}")
            
            # Run multiple iterations with different random inputs
            iteration_results = []
            for i in range(num_iterations):
                # Generate random inputs
                torch.manual_seed(42 + i)  # Deterministic but different each iteration
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + i)
                
                hidden_states = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=self.device, dtype=torch.float32
                )
                
                attention_mask = torch.ones(
                    batch_size, seq_len,
                    device=self.device, dtype=torch.bool
                )
                
                position_ids = torch.arange(
                    seq_len, device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
                
                past_key_values = None
                
                inputs = [hidden_states, attention_mask, position_ids, past_key_values]
                
                # Determine graph type
                graph_type = None
                if graph_type_mapper:
                    graph_type = graph_type_mapper(seq_len)
                
                # Validate
                result = self.validate_single_forward(
                    eager_func=eager_func,
                    graph_func=graph_func,
                    inputs=inputs,
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    graph_type=graph_type
                )
                
                iteration_results.append(result)
            
            # Combine iteration results
            if iteration_results:
                # Use the last result as representative, but mark if any failed
                representative = iteration_results[-1]
                any_failed = any(not r.passed for r in iteration_results)
                
                if any_failed:
                    representative.passed = False
                    representative.warnings.append(
                        f"Failed in {sum(1 for r in iteration_results if not r.passed)}/"
                        f"{len(iteration_results)} iterations"
                    )
                
                results[seq_len] = representative
        
        return results
    
    def validate_model_wrapper(
        self,
        model_wrapper,
        context_lengths: List[int] = None,
        batch_size: int = 1,
        vocab_size: int = 32000,
        num_iterations: int = 2
    ) -> Dict[int, ValidationResult]:
        """
        Validate a wrapped model with CUDA graphs.
        
        Args:
            model_wrapper: Wrapped model with CUDA graph support
            context_lengths: List of sequence lengths to test
            batch_size: Batch size for validation
            vocab_size: Vocabulary size for input IDs
            num_iterations: Number of iterations per context length
            
        Returns:
            Dictionary mapping context lengths to validation results
        """
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096]
        
        results = {}
        
        for seq_len in context_lengths:
            logger.info(f"Validating model wrapper at context length: {seq_len}")
            
            iteration_results = []
            for i in range(num_iterations):
                # Generate random inputs
                torch.manual_seed(42 + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + i)
                
                input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len),
                    device=self.device, dtype=torch.long
                )
                
                attention_mask = torch.ones(
                    batch_size, seq_len,
                    device=self.device, dtype=torch.bool
                )
                
                # Run eager forward (bypassing wrapper if possible)
                # Note: This assumes the wrapper has an eager_forward method
                if hasattr(model_wrapper, 'eager_forward'):
                    eager_outputs = model_wrapper.eager_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                else:
                    # Fallback: temporarily disable graphs
                    if hasattr(model_wrapper, 'graph_manager'):
                        original_state = model_wrapper.graph_manager.enable_graphs
                        model_wrapper.graph_manager.enable_graphs = False
                    
                    eager_outputs = model_wrapper.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    if hasattr(model_wrapper, 'graph_manager'):
                        model_wrapper.graph_manager.enable_graphs = original_state
                
                # Run graph forward
                graph_outputs = model_wrapper.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Compare outputs
                comparison = self._compare_outputs(eager_outputs, graph_outputs)
                
                # Create validation result
                result = ValidationResult(
                    passed=comparison["all_passed"],
                    max_absolute_error=comparison["max_absolute_error"],
                    max_relative_error=comparison["max_relative_error"],
                    mean_absolute_error=comparison["mean_absolute_error"],
                    mean_relative_error=comparison["mean_relative_error"],
                    num_tensors_compared=comparison["num_tensors_compared"],
                    num_elements_compared=comparison["num_elements_compared"],
                    validation_time_ms=0,  # Not measured separately
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    graph_type=f"seq_len_{seq_len}",
                    error_details=comparison["results"] if not comparison["all_passed"] else None
                )
                
                iteration_results.append(result)
            
            # Combine iteration results
            if iteration_results:
                representative = iteration_results[-1]
                any_failed = any(not r.passed for r in iteration_results)
                
                if any_failed:
                    representative.passed = False
                    representative.warnings.append(
                        f"Failed in {sum(1 for r in iteration_results if not r.passed)}/"
                        f"{len(iteration_results)} iterations"
                    )
                
                results[seq_len] = representative
        
        return results
    
    def generate_validation_report(
        self,
        validation_results: Dict[int, ValidationResult],
        title: str = "CUDA Graph Output Validation Report"
    ) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Dictionary of validation results
            title: Report title
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(title)
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        total_tests = len(validation_results)
        passed_tests = sum(1 for r in validation_results.values() if r.passed)
        failed_tests = total_tests - passed_tests
        
        report_lines.append(f"SUMMARY")
        report_lines.append(f"  Total tests: {total_tests}")
        report_lines.append(f"  Passed: {passed_tests}")
        report_lines.append(f"  Failed: {failed_tests}")
        report_lines.append(f"  Pass rate: {passed_tests/total_tests*100:.1f}%")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 80)
        
        for seq_len, result in validation_results.items():
            status = "PASS" if result.passed else "FAIL"
            report_lines.append(f"Context Length: {seq_len} [{status}]")
            report_lines.append(f"  Batch Size: {result.batch_size}")
            report_lines.append(f"  Graph Type: {result.graph_type or 'N/A'}")
            report_lines.append(f"  Max Absolute Error: {result.max_absolute_error:.2e}")
            report_lines.append(f"  Max Relative Error: {result.max_relative_error:.2e}")
            report_lines.append(f"  Tensors Compared: {result.num_tensors_compared}")
            report_lines.append(f"  Elements Compared: {result.num_elements_compared:,}")
            
            if result.warnings:
                report_lines.append(f"  Warnings: {', '.join(result.warnings)}")
            
            if not result.passed and result.error_details:
                report_lines.append(f"  Errors:")
                for error in result.error_details[:3]:  # Show first 3 errors
                    if isinstance(error, dict) and "error" in error:
                        report_lines.append(f"    - {error['error']}")
            
            report_lines.append("")
        
        # Tolerance information
        report_lines.append("VALIDATION CONFIGURATION")
        report_lines.append("-" * 80)
        report_lines.append(f"  Absolute Tolerance: {self.absolute_tolerance}")
        report_lines.append(f"  Relative Tolerance: {self.relative_tolerance}")
        report_lines.append(f"  NaN Check: {'Enabled' if self.enable_nan_check else 'Disabled'}")
        report_lines.append(f"  Inf Check: {'Enabled' if self.enable_inf_check else 'Disabled'}")
        report_lines.append(f"  Device: {self.device}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_validation_results(
        self,
        validation_results: Dict[int, ValidationResult],
        output_dir: Path,
        filename: str = "validation_results.json"
    ):
        """
        Save validation results to disk.
        
        Args:
            validation_results: Dictionary of validation results
            output_dir: Directory to save results
            filename: Output filename
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        # Convert to serializable format
        serializable_results = {
            str(seq_len): result.to_dict()
            for seq_len, result in validation_results.items()
        }
        
        # Add metadata
        metadata = {
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "device": self.device
        }
        
        data = {
            "metadata": metadata,
            "results": serializable_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Validation results saved to: {output_path}")
        
        # Also save human-readable report
        report = self.generate_validation_report(validation_results)
        report_path = output_dir / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to: {report_path}")


def create_default_validator() -> OutputValidator:
    """Create a validator with default settings for CUDA graph validation."""
    return OutputValidator(
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
        enable_nan_check=True,
        enable_inf_check=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )