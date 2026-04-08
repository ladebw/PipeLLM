#!/usr/bin/env python3
"""
Task 2.6: Validate output correctness vs baseline

WARNING: SIMULATED DATA — not from real hardware
This script validates output correctness of async prefetch infrastructure
vs eager baseline. All comparisons are based on simulated data and must be
validated on real CUDA hardware before deployment (see ROADMAP task 2.6).

Validates:
1. Async prefetch vs eager baseline output equivalence
2. Stream synchronization correctness
3. Memory consistency under concurrent access
4. Stress test reliability
5. Integration with existing validation infrastructure
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline_parallel.async_output_validation import (
    AsyncOutputValidator,
    create_async_validator,
    AsyncValidationMode
)


def create_task_summary(results: dict, output_dir: str, execution_time: float) -> dict:
    """
    Create comprehensive task summary.
    
    Args:
        results: Validation results dictionary
        output_dir: Output directory path
        execution_time: Total execution time in seconds
    
    Returns:
        Task summary dictionary
    """
    # Calculate overall statistics
    total_modes = len(results)
    passed_modes = sum(1 for r in results.values() if r.passed)
    failed_modes = total_modes - passed_modes
    
    total_tests = sum(r.comparison_count for r in results.values())
    passed_tests = sum(r.passed_count for r in results.values())
    failed_tests = sum(r.failed_count for r in results.values())
    
    # Check for specific issues
    stream_sync_issues = any(
        hasattr(r, 'stream_sync_correct') and not r.stream_sync_correct 
        for r in results.values()
    )
    
    memory_consistency_issues = any(
        hasattr(r, 'memory_consistent') and not r.memory_consistent
        for r in results.values()
    )
    
    race_conditions = sum(
        r.race_conditions_detected for r in results.values() 
        if hasattr(r, 'race_conditions_detected')
    )
    
    summary = {
        "task": "2.6",
        "title": "Validate output correctness vs baseline",
        "timestamp": time.time(),
        "execution_time_seconds": execution_time,
        "overall_status": "PASS" if failed_modes == 0 else "FAIL",
        "statistics": {
            "validation_modes": {
                "total": total_modes,
                "passed": passed_modes,
                "failed": failed_modes,
                "pass_rate": passed_modes / total_modes * 100 if total_modes > 0 else 0
            },
            "individual_tests": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": passed_tests / total_tests * 100 if total_tests > 0 else 0
            },
            "issues_detected": {
                "stream_sync_issues": stream_sync_issues,
                "memory_consistency_issues": memory_consistency_issues,
                "race_conditions": race_conditions,
                "stress_test_failures": any(
                    hasattr(r, 'stress_test_passed') and not r.stress_test_passed
                    for r in results.values()
                )
            }
        },
        "validation_modes": {},
        "output_files": [],
        "recommendations": [],
        "notes": [
            "WARNING: These results are from CPU-based simulation",
            "Real CUDA hardware validation required before deployment",
            "See ROADMAP task 2.6 for hardware validation requirements",
            "Integration with Phase 1 validation infrastructure complete"
        ]
    }
    
    # Add details for each validation mode
    for mode_name, result in results.items():
        summary["validation_modes"][mode_name] = {
            "status": "PASS" if result.passed else "FAIL",
            "mode": result.validation_mode.value if hasattr(result, 'validation_mode') else "unknown",
            "tests": {
                "total": result.comparison_count,
                "passed": result.passed_count,
                "failed": result.failed_count
            },
            "details": {
                "stream_sync_correct": getattr(result, 'stream_sync_correct', True),
                "memory_consistent": getattr(result, 'memory_consistent', True),
                "race_conditions": getattr(result, 'race_conditions_detected', 0),
                "stress_test_passed": getattr(result, 'stress_test_passed', True),
                "async_overhead_ms": getattr(result, 'async_overhead_ms', 0),
                "overlap_percent": getattr(result, 'memory_transfer_overlap_percent', 0)
            }
        }
    
    # List output files
    output_path = Path(output_dir)
    if output_path.exists():
        for file_path in output_path.glob("*"):
            if file_path.is_file():
                try:
                    rel_path = str(file_path.relative_to(project_root))
                except ValueError:
                    rel_path = str(file_path)
                
                summary["output_files"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "path": rel_path
                })
    
    # Generate recommendations based on results
    if failed_modes > 0:
        summary["recommendations"].append(
            "Review failed validation modes and fix implementation issues"
        )
    
    if stream_sync_issues:
        summary["recommendations"].append(
            "Fix stream synchronization issues in async prefetch implementation"
        )
    
    if memory_consistency_issues:
        summary["recommendations"].append(
            "Address memory consistency issues in buffer management"
        )
    
    if race_conditions > 0:
        summary["recommendations"].append(
            f"Fix {race_conditions} race condition(s) detected in validation"
        )
    
    # Always recommend hardware validation
    summary["recommendations"].append(
        "Run validation on real CUDA hardware to confirm results"
    )
    
    return summary


def save_task_summary(summary: dict, output_dir: str):
    """
    Save task summary to JSON file.
    
    Args:
        summary: Task summary dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir) / "task_2_6_summary.json"
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Task summary saved to: {output_path}")


def print_results_summary(results: dict, execution_time: float):
    """
    Print human-readable results summary.
    
    Args:
        results: Validation results dictionary
        execution_time: Total execution time in seconds
    """
    print("\n" + "=" * 80)
    print("TASK 2.6: ASYNC PREFETCH OUTPUT VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"\nWARNING: SIMULATED DATA — not from real hardware")
    print("These results are from CPU-based simulation and must be validated")
    print("on real CUDA hardware before deployment (see ROADMAP task 2.6).\n")
    
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    # Calculate statistics
    total_modes = len(results)
    passed_modes = sum(1 for r in results.values() if r.passed)
    failed_modes = total_modes - passed_modes
    
    total_tests = sum(r.comparison_count for r in results.values())
    passed_tests = sum(r.passed_count for r in results.values())
    failed_tests = sum(r.failed_count for r in results.values())
    
    print(f"\nOverall Status: {'PASS' if failed_modes == 0 else 'FAIL'}")
    print(f"Validation Modes: {passed_modes} passed, {failed_modes} failed")
    print(f"Individual Tests: {passed_tests} passed, {failed_tests} failed")
    print(f"Test Pass Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Check for specific issues
    stream_sync_issues = any(
        hasattr(r, 'stream_sync_correct') and not r.stream_sync_correct 
        for r in results.values()
    )
    
    memory_consistency_issues = any(
        hasattr(r, 'memory_consistent') and not r.memory_consistent
        for r in results.values()
    )
    
    race_conditions = sum(
        r.race_conditions_detected for r in results.values() 
        if hasattr(r, 'race_conditions_detected')
    )
    
    if stream_sync_issues or memory_consistency_issues or race_conditions > 0:
        print(f"\nIssues Detected:")
        if stream_sync_issues:
            print("  - Stream synchronization issues")
        if memory_consistency_issues:
            print("  - Memory consistency issues")
        if race_conditions > 0:
            print(f"  - {race_conditions} race condition(s)")
    
    # Detailed results by mode
    print(f"\nDetailed Results by Validation Mode:")
    print("-" * 40)
    
    for mode_name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        mode_type = result.validation_mode.value if hasattr(result, 'validation_mode') else "unknown"
        
        print(f"\n{mode_name}: [{status}]")
        print(f"  Type: {mode_type}")
        print(f"  Tests: {result.passed_count} passed, {result.failed_count} failed")
        
        if hasattr(result, 'stream_sync_correct'):
            print(f"  Stream Sync: {'OK' if result.stream_sync_correct else 'ISSUES'}")
        
        if hasattr(result, 'memory_consistent'):
            print(f"  Memory Consistency: {'OK' if result.memory_consistent else 'ISSUES'}")
        
        if hasattr(result, 'race_conditions_detected') and result.race_conditions_detected > 0:
            print(f"  Race Conditions: {result.race_conditions_detected}")
        
        if hasattr(result, 'async_overhead_ms') and result.async_overhead_ms > 0:
            print(f"  Async Overhead: {result.async_overhead_ms:.2f} ms")
        
        if hasattr(result, 'memory_transfer_overlap_percent'):
            print(f"  Memory Overlap: {result.memory_transfer_overlap_percent:.1f}%")
    
    print("\n" + "=" * 80)
    print("END OF RESULTS")
    print("=" * 80)


def main():
    """Main execution function."""
    print("Starting Task 2.6: Validate output correctness vs baseline")
    print("=" * 80)
    
    # Configuration
    output_dir = "phase2_results/task_2_6"
    enable_race_testing = False  # Set to True to test with race conditions
    stress_iterations = 50
    
    print(f"Output directory: {output_dir}")
    print(f"Race condition testing: {'ENABLED' if enable_race_testing else 'DISABLED'}")
    print(f"Stress test iterations: {stress_iterations}")
    print("\nWARNING: Running CPU-based simulation (no real CUDA hardware)")
    print("Real hardware validation required for production deployment.\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create validator
    validator = create_async_validator(
        enable_race_condition_testing=enable_race_testing,
        stress_test_iterations=stress_iterations
    )
    
    # Create test input tensor
    print("Creating test input tensor...")
    input_tensor = torch.randn(256, dtype=torch.float32)
    print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    # Run validation
    start_time = time.time()
    print("\nRunning async prefetch output validation...")
    
    try:
        results = validator.validate_all_modes(
            input_tensor=input_tensor,
            output_dir=output_dir
        )
        
        execution_time = time.time() - start_time
        print(f"Validation completed in {execution_time:.2f} seconds")
        
        # Create and save task summary
        summary = create_task_summary(results, output_dir, execution_time)
        save_task_summary(summary, output_dir)
        
        # Print results summary
        print_results_summary(results, execution_time)
        
        # Return appropriate exit code
        if summary["overall_status"] == "FAIL":
            print("\n[WARNING] Validation failed! Review recommendations above.")
            return 1
        else:
            print("\n[OK] All validation tests passed (simulated environment).")
            print("     Note: Real CUDA hardware validation still required.")
            return 0
            
    except Exception as e:
        print(f"\n[ERROR] Error during output validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())