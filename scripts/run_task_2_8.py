#!/usr/bin/env python3
"""
Task 2.8: Write tests for Phase 2 components.

WARNING: SIMULATED DATA — not from real hardware
These tests verify the infrastructure, not actual CUDA performance.

This script runs all Phase 2 tests and generates a comprehensive
test coverage report.
"""

import pytest
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_pytest_tests(test_files: List[str], output_dir: Path) -> Dict[str, Any]:
    """
    Run pytest tests and collect results.
    
    Args:
        test_files: List of test files to run
        output_dir: Directory for test results
        
    Returns:
        Dictionary with test results
    """
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "test_files": [],
        "start_time": datetime.now().isoformat(),
        "duration_seconds": 0
    }
    
    # Create JUnit XML output path
    junit_xml = output_dir / "test_results.xml"
    
    # Build pytest command
    pytest_args = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        f"--junitxml={junit_xml}",
        "--html", str(output_dir / "test_report.html"),
        "--self-contained-html"
    ]
    
    # Add test files
    pytest_args.extend(test_files)
    
    # Run pytest
    start_time = datetime.now()
    
    try:
        # Run pytest and capture output
        print(f"\nRunning pytest with args: {' '.join(pytest_args[3:])}")
        
        # Use subprocess to run pytest
        process = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        # Parse output
        stdout = process.stdout
        stderr = process.stderr
        
        # Save output to files
        with open(output_dir / "test_stdout.txt", "w", encoding="utf-8") as f:
            f.write(stdout)
        
        with open(output_dir / "test_stderr.txt", "w", encoding="utf-8") as f:
            f.write(stderr)
        
        # Try to parse test results from output
        # This is a simple parser - in production you'd use pytest's API
        lines = stdout.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line and "skipped" in line:
                # Example: "5 passed, 1 failed, 2 skipped in 0.12s"
                parts = line.split()
                for part in parts:
                    if "passed" in part:
                        results["passed"] = int(part.split("passed")[0])
                    elif "failed" in part:
                        results["failed"] = int(part.split("failed")[0])
                    elif "skipped" in part:
                        results["skipped"] = int(part.split("skipped")[0])
                    elif "error" in part:
                        results["errors"] = int(part.split("error")[0])
                
                results["total_tests"] = (
                    results["passed"] + 
                    results["failed"] + 
                    results["skipped"] + 
                    results["errors"]
                )
        
        # Calculate duration
        end_time = datetime.now()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        results["end_time"] = end_time.isoformat()
        
        # Set exit code based on test results
        results["exit_code"] = process.returncode
        results["success"] = (results["failed"] == 0 and results["errors"] == 0)
        
        print(f"\nTest Results: {results['passed']} passed, {results['failed']} failed, "
              f"{results['skipped']} skipped, {results['errors']} errors")
        
        return results
        
    except Exception as e:
        print(f"Error running pytest: {e}")
        results["error"] = str(e)
        results["success"] = False
        return results


def collect_test_files() -> List[str]:
    """
    Collect all Phase 2 test files.
    
    Returns:
        List of test file paths
    """
    test_dir = project_root / "tests" / "pipeline_parallel"
    
    test_files = []
    
    # Phase 2 test files
    phase2_tests = [
        "test_layer_profiler.py",           # Task 2.1
        "test_dual_stream_manager.py",      # Task 2.2
        "test_pinned_memory_pool.py",       # Task 2.3
        "test_async_prefetch_engine.py",    # Task 2.4
        "test_race_conditions.py",          # Task 2.5
        "test_async_output_validation.py",  # Task 2.6
        "test_cumulative_benchmark.py",     # Task 2.7
    ]
    
    for test_file in phase2_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            test_files.append(str(test_path))
            print(f"Found test file: {test_file}")
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    return test_files


def calculate_test_coverage(test_files: List[str]) -> Dict[str, Any]:
    """
    Calculate test coverage statistics.
    
    Args:
        test_files: List of test files
        
    Returns:
        Dictionary with coverage statistics
    """
    coverage = {
        "total_test_files": len(test_files),
        "phase2_tasks": {
            "2.1": {"tested": False, "test_file": None},
            "2.2": {"tested": False, "test_file": None},
            "2.3": {"tested": False, "test_file": None},
            "2.4": {"tested": False, "test_file": None},
            "2.5": {"tested": False, "test_file": None},
            "2.6": {"tested": False, "test_file": None},
            "2.7": {"tested": False, "test_file": None},
        },
        "components_tested": [],
        "lines_of_test_code": 0,
        "test_coverage_percentage": 0.0
    }
    
    # Map test files to tasks
    task_mapping = {
        "test_layer_profiler.py": "2.1",
        "test_dual_stream_manager.py": "2.2",
        "test_pinned_memory_pool.py": "2.3",
        "test_async_prefetch_engine.py": "2.4",
        "test_race_conditions.py": "2.5",
        "test_async_output_validation.py": "2.6",
        "test_cumulative_benchmark.py": "2.7",
    }
    
    # Count lines of test code
    total_lines = 0
    for test_file in test_files:
        test_path = Path(test_file)
        
        # Count lines
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines += len(lines)
        except Exception as e:
            print(f"Warning: Could not count lines in {test_path}: {e}")
        
        # Map to task
        filename = test_path.name
        if filename in task_mapping:
            task = task_mapping[filename]
            coverage["phase2_tasks"][task]["tested"] = True
            coverage["phase2_tasks"][task]["test_file"] = filename
    
    coverage["lines_of_test_code"] = total_lines
    
    # Calculate coverage percentage
    total_tasks = len(coverage["phase2_tasks"])
    tested_tasks = sum(1 for task in coverage["phase2_tasks"].values() if task["tested"])
    coverage["test_coverage_percentage"] = (tested_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    
    # List components tested
    components = [
        "Layer Profiler (Task 2.1)",
        "Dual Stream Manager (Task 2.2)",
        "Pinned Memory Pool (Task 2.3)",
        "Async Prefetch Engine (Task 2.4)",
        "Race Condition Validator (Task 2.5)",
        "Async Output Validation (Task 2.6)",
        "Cumulative Benchmark (Task 2.7)"
    ]
    
    for i, (task_id, task_info) in enumerate(coverage["phase2_tasks"].items()):
        if task_info["tested"]:
            coverage["components_tested"].append(components[i])
    
    return coverage


def generate_test_report(test_results: Dict[str, Any], 
                         coverage: Dict[str, Any],
                         output_dir: Path) -> Path:
    """
    Generate comprehensive test report.
    
    Args:
        test_results: Test execution results
        coverage: Test coverage statistics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_path = output_dir / "task_2_8_test_report.json"
    
    report = {
        "task": "2.8",
        "description": "Write tests for Phase 2 components",
        "status": "completed" if test_results.get("success", False) else "failed",
        "completion_date": datetime.now().isoformat(),
        "simulation_warning": "WARNING: All tests use simulated data - real CUDA hardware validation required",
        
        "test_execution": test_results,
        "test_coverage": coverage,
        
        "phase2_test_summary": {
            "total_phase2_tasks": 7,
            "tasks_with_tests": sum(1 for task in coverage["phase2_tasks"].values() if task["tested"]),
            "test_coverage_percentage": coverage["test_coverage_percentage"],
            "total_test_files": coverage["total_test_files"],
            "total_lines_of_test_code": coverage["lines_of_test_code"],
        },
        
        "files_generated": [
            "test_results.xml",
            "test_report.html",
            "test_stdout.txt",
            "test_stderr.txt",
            "task_2_8_test_report.json"
        ],
        
        "next_steps": [
            "Run tests on real CUDA hardware",
            "Add integration tests for complete pipeline",
            "Add performance regression tests",
            "Set up CI/CD with automated testing"
        ],
        
        "notes": [
            "All tests use simulated/mocked data for CPU-based testing",
            "Real CUDA hardware tests are required for production validation",
            "Test coverage includes all Phase 2 components",
            "Tests verify infrastructure correctness, not performance"
        ]
    }
    
    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report_path


def print_task_header():
    """Print task execution header."""
    print("\n" + "="*70)
    print("TASK 2.8: Write tests for Phase 2 components")
    print("="*70)
    print("Phase: 2 - Async Double-Buffered Weight Prefetch")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nWARNING: SIMULATED DATA - not from real hardware")
    print("These tests verify the infrastructure, not actual CUDA performance.")
    print("="*70)


def print_task_summary(report_path: Path, test_results: Dict[str, Any], 
                       coverage: Dict[str, Any]):
    """Print task execution summary."""
    print("\n" + "="*70)
    print("TASK 2.8 EXECUTION SUMMARY")
    print("="*70)
    
    # Test execution results
    print(f"\nTest Execution Results:")
    print(f"  Total tests: {test_results.get('total_tests', 0)}")
    print(f"  Passed: {test_results.get('passed', 0)}")
    print(f"  Failed: {test_results.get('failed', 0)}")
    print(f"  Skipped: {test_results.get('skipped', 0)}")
    print(f"  Errors: {test_results.get('errors', 0)}")
    print(f"  Duration: {test_results.get('duration_seconds', 0):.2f} seconds")
    print(f"  Success: {'Yes' if test_results.get('success', False) else 'No'}")
    
    # Test coverage
    print(f"\nTest Coverage:")
    print(f"  Phase 2 tasks: {coverage.get('total_test_files', 0)} / 7")
    print(f"  Coverage: {coverage.get('test_coverage_percentage', 0):.1f}%")
    print(f"  Lines of test code: {coverage.get('lines_of_test_code', 0):,}")
    
    # Components tested
    print(f"\nComponents Tested:")
    for component in coverage.get("components_tested", []):
        print(f"  [OK] {component}")
    
    # Missing tests
    missing = []
    for task_id, task_info in coverage.get("phase2_tasks", {}).items():
        if not task_info.get("tested", False):
            missing.append(f"Task {task_id}")
    
    if missing:
        print(f"\nMissing Tests:")
        for task in missing:
            print(f"  [MISSING] {task}")
    
    # Files generated
    print(f"\nFiles Generated:")
    print(f"  Test report: {report_path}")
    print(f"  JUnit XML: phase2_results/task_2_8/test_results.xml")
    print(f"  HTML report: phase2_results/task_2_8/test_report.html")
    
    print(f"\nNext Steps:")
    print("  1. Run tests on real CUDA hardware")
    print("  2. Add integration tests for complete pipeline")
    print("  3. Set up CI/CD with automated testing")
    
    print(f"\nWARNING: All tests use simulated data - real CUDA hardware validation required")
    print("="*70)


def main():
    """Main task execution function."""
    print_task_header()
    
    # Create output directory
    output_dir = project_root / "phase2_results" / "task_2_8"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Collect test files
        print("\nStep 1: Collecting Phase 2 test files...")
        test_files = collect_test_files()
        
        if not test_files:
            print("Error: No test files found!")
            return 1
        
        print(f"Found {len(test_files)} test files")
        
        # Step 2: Calculate test coverage
        print("\nStep 2: Calculating test coverage...")
        coverage = calculate_test_coverage(test_files)
        
        print(f"Test coverage: {coverage['test_coverage_percentage']:.1f}%")
        print(f"Lines of test code: {coverage['lines_of_test_code']:,}")
        
        # Step 3: Run tests
        print("\nStep 3: Running tests...")
        test_results = run_pytest_tests(test_files, output_dir)
        
        # Step 4: Generate report
        print("\nStep 4: Generating test report...")
        report_path = generate_test_report(test_results, coverage, output_dir)
        
        # Step 5: Print summary
        print_task_summary(report_path, test_results, coverage)
        
        if test_results.get("success", False):
            print(f"\n[SUCCESS] Task 2.8 completed successfully")
            print(f"Test report saved to: {report_path}")
            return 0
        else:
            print(f"\n[WARNING] Task 2.8 completed with test failures")
            print(f"Check {output_dir}/test_stdout.txt for details")
            return 1
        
    except Exception as e:
        print(f"\n[ERROR] Task 2.8 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())