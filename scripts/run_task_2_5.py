#!/usr/bin/env python3
"""
Task 2.5: Validate no race conditions under stress test

WARNING: SIMULATED DATA — not from real hardware
This script runs CPU-based race condition validation to detect potential
synchronization issues in the async prefetch infrastructure. Real CUDA
hardware validation is required before deployment (see ROADMAP task 2.5).

Validates:
1. Stream synchronization correctness between compute and copy streams
2. Memory access ordering for shared buffers
3. Double-buffer swap safety under concurrent access
4. Concurrent read/write patterns
5. Stress test reliability under high load
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline_parallel.race_condition_validator import validate_race_conditions


def create_task_summary(results: dict, output_dir: str) -> dict:
    """
    Create comprehensive task summary.
    
    Args:
        results: Validation results dictionary
        output_dir: Output directory path
    
    Returns:
        Task summary dictionary
    """
    summary = {
        "task": "2.5",
        "title": "Validate no race conditions under stress test",
        "timestamp": time.time(),
        "status": results.get("status", "UNKNOWN"),
        "execution_time": results.get("metrics", {}).get("total_execution_time", 0),
        "test_results": {
            "total": results.get("metrics", {}).get("total_tests", 0),
            "passed": results.get("metrics", {}).get("passed_tests", 0),
            "failed": results.get("metrics", {}).get("failed_tests", 0),
            "critical_issues": results.get("metrics", {}).get("critical_issues", 0),
            "high_severity": results.get("metrics", {}).get("high_severity", 0),
            "medium_severity": results.get("metrics", {}).get("medium_severity", 0),
            "low_severity": results.get("metrics", {}).get("low_severity", 0),
        },
        "output_files": [],
        "recommendations": results.get("recommendations", []),
        "notes": [
            "WARNING: These results are from CPU-based simulation",
            "Real CUDA hardware validation required before deployment",
            "See ROADMAP task 2.5 for hardware validation requirements"
        ]
    }
    
    # List output files
    output_path = Path(output_dir)
    if output_path.exists():
        for file_path in output_path.glob("*"):
            if file_path.is_file():
                try:
                    rel_path = str(file_path.relative_to(project_root))
                except ValueError:
                    # If not relative to project root, use absolute path
                    rel_path = str(file_path)
                
                summary["output_files"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "path": rel_path
                })
    
    return summary


def save_task_summary(summary: dict, output_dir: str):
    """
    Save task summary to JSON file.
    
    Args:
        summary: Task summary dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir) / "task_2_5_summary.json"
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Task summary saved to: {output_path}")


def print_results_summary(results: dict):
    """
    Print human-readable results summary.
    
    Args:
        results: Validation results dictionary
    """
    print("\n" + "=" * 80)
    print("TASK 2.5: RACE CONDITION VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"\nWARNING: SIMULATED DATA — not from real hardware")
    print("These results are from CPU-based simulation and must be validated")
    print("on real CUDA hardware before deployment (see ROADMAP task 2.5).\n")
    
    metrics = results.get("metrics", {})
    status = results.get("status", "UNKNOWN")
    
    print(f"Overall Status: {status}")
    print(f"Execution Time: {metrics.get('total_execution_time', 0):.2f} seconds")
    print(f"Pass Rate: {metrics.get('pass_rate', 0):.1f}%")
    
    print(f"\nTest Results:")
    print(f"  Total Tests: {metrics.get('total_tests', 0)}")
    print(f"  Passed: {metrics.get('passed_tests', 0)}")
    print(f"  Failed: {metrics.get('failed_tests', 0)}")
    
    if metrics.get('failed_tests', 0) > 0:
        print(f"\nIssues Detected:")
        print(f"  Critical: {metrics.get('critical_issues', 0)}")
        print(f"  High Severity: {metrics.get('high_severity', 0)}")
        print(f"  Medium Severity: {metrics.get('medium_severity', 0)}")
        print(f"  Low Severity: {metrics.get('low_severity', 0)}")
    
    # Print individual test results
    print(f"\nDetailed Test Results:")
    print("-" * 40)
    for test_result in results.get("results", []):
        test_name = test_result.get("test_name", "Unknown")
        detected = test_result.get("detected", False)
        severity = test_result.get("severity", "unknown")
        status = "FAIL" if detected else "PASS"
        
        print(f"\n{test_name} [{status}]")
        print(f"  Type: {test_result.get('condition_type', 'unknown')}")
        print(f"  Severity: {severity}")
        
        if detected:
            print(f"  Issue: {test_result.get('description', 'No description')}")
    
    # Print recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    print("\n" + "=" * 80)
    print("END OF RESULTS")
    print("=" * 80)


def main():
    """Main execution function."""
    print("Starting Task 2.5: Validate no race conditions under stress test")
    print("=" * 80)
    
    # Configuration
    output_dir = "phase2_results/task_2_5"
    max_threads = 10
    stress_iterations = 1000
    
    print(f"Output directory: {output_dir}")
    print(f"Max concurrent threads: {max_threads}")
    print(f"Stress test iterations: {stress_iterations}")
    print("\nWARNING: Running CPU-based simulation (no real CUDA hardware)")
    print("Real hardware validation required for production deployment.\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run race condition validation
    start_time = time.time()
    print("Running race condition validation tests...")
    
    try:
        results = validate_race_conditions(
            output_dir=output_dir,
            max_threads=max_threads,
            stress_iterations=stress_iterations
        )
        
        execution_time = time.time() - start_time
        print(f"Validation completed in {execution_time:.2f} seconds")
        
        # Create and save task summary
        summary = create_task_summary(results, output_dir)
        save_task_summary(summary, output_dir)
        
        # Print results summary
        print_results_summary(results)
        
        # Return appropriate exit code
        if results.get("status") == "FAIL":
            print("\n[WARNING] Race conditions detected! Review recommendations above.")
            return 1
        else:
            print("\n[OK] All race condition tests passed (simulated environment).")
            print("     Note: Real CUDA hardware validation still required.")
            return 0
            
    except Exception as e:
        print(f"\n[ERROR] Error during race condition validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())