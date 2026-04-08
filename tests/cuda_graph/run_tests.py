"""
Test runner for CUDA Graph Compilation module.

This script runs all tests for the CUDA graph implementation
and generates a comprehensive test report.
"""

import sys
import os
import pytest
import torch
import json
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def run_tests() -> Dict[str, Any]:
    """
    Run all CUDA graph tests and return results.
    
    Returns:
        Dictionary containing test results and statistics
    """
    print("=" * 70)
    print("CUDA GRAPH COMPILATION - TEST SUITE")
    print("Phase 1, Task 1.3: Implement static CUDA graph capture")
    print("=" * 70)
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"Device: {device_name}")
        print(f"CUDA Version: {cuda_version}")
    else:
        print("WARNING: CUDA not available - GPU tests will be skipped")
    
    print()
    
    # Test files to run (all Phase 1 tests)
    test_files = [
        'test_cuda_graph_capture.py',      # Task 1.3
        'test_bucket_management.py',       # Task 1.4
        'test_output_validation.py',       # Task 1.5
        'test_benchmarking.py',           # Task 1.6
        'test_integration.py',            # Integration tests
        'test_performance.py',            # Performance tests
        'test_llama_integration.py',      # Model integration
    ]
    
    # Run tests
    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        print(f"Running tests from: {test_file}")
        print("-" * 50)
        
        # Run pytest
        pytest_args = [
            test_path,
            '-v',
            '--tb=short',
            '-q',  # Quiet mode for summary
        ]
        
        # Capture output
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exit_code = pytest.main(pytest_args)
        
        # Parse output
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        # Count results from output
        lines = output.split('\n')
        
        for line in lines:
            if 'passed' in line and 'failed' in line and 'skipped' in line:
                # Parse summary line like: "3 passed, 1 failed, 2 skipped in 0.12s"
                parts = line.split()
                for part in parts:
                    if 'passed' in part:
                        total_passed += int(part.split('passed')[0])
                    elif 'failed' in part:
                        total_failed += int(part.split('failed')[0])
                    elif 'skipped' in part:
                        total_skipped += int(part.split('skipped')[0])
        
        # Store results
        file_results = {
            'file': test_file,
            'exit_code': exit_code,
            'output': output,
            'error': error_output,
        }
        all_results.append(file_results)
        
        print(f"  Exit code: {exit_code}")
        print()
    
    # Generate summary
    total_tests = total_passed + total_failed + total_skipped
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'cuda_available': cuda_available,
        'device_name': device_name if cuda_available else None,
        'cuda_version': cuda_version if cuda_available else None,
        'total_tests': total_tests,
        'passed': total_passed,
        'failed': total_failed,
        'skipped': total_skipped,
        'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
        'test_files': len(test_files),
        'all_passed': total_failed == 0,
    }
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({summary['success_rate']:.1f}%)")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    print(f"Test Files: {len(test_files)}")
    print()
    
    if total_failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {total_failed} TEST(S) FAILED")
    
    print()
    
    # Check CUDA-specific requirements
    if cuda_available:
        print("CUDA-SPECIFIC CHECKS:")
        print("-" * 30)
        
        # Check PyTorch CUDA version
        print(f"PyTorch CUDA: {torch.version.cuda}")
        
        # Check device properties
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
        
        # Check if compute capability supports CUDA graphs
        min_compute_capability = 7.0  # Volta+
        device_compute_capability = float(f"{props.major}.{props.minor}")
        
        if device_compute_capability >= min_compute_capability:
            print(f"✅ Compute capability supports CUDA graphs ({device_compute_capability} >= {min_compute_capability})")
        else:
            print(f"⚠ Compute capability may not fully support CUDA graphs ({device_compute_capability} < {min_compute_capability})")
    
    print()
    print("=" * 70)
    
    # Save detailed report
    save_test_report(summary, all_results)
    
    return summary


def save_test_report(summary: Dict[str, Any], all_results: List[Dict[str, Any]]):
    """Save test results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    report = {
        'summary': summary,
        'detailed_results': all_results,
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Detailed test report saved to: {report_file}")
    
    # Also save a human-readable summary
    txt_report_file = f"test_summary_{timestamp}.txt"
    
    with open(txt_report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CUDA GRAPH COMPILATION - TEST REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Test Date: {summary['timestamp']}\n")
        f.write(f"CUDA Available: {summary['cuda_available']}\n")
        
        if summary['cuda_available']:
            f.write(f"Device: {summary['device_name']}\n")
            f.write(f"CUDA Version: {summary['cuda_version']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Tests: {summary['total_tests']}\n")
        f.write(f"Passed: {summary['passed']}\n")
        f.write(f"Failed: {summary['failed']}\n")
        f.write(f"Skipped: {summary['skipped']}\n")
        f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
        f.write(f"All Tests Passed: {summary['all_passed']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        if summary['all_passed']:
            f.write("✅ All tests passed successfully.\n")
            f.write("✅ CUDA graph implementation is working correctly.\n")
            f.write("✅ Ready for integration with actual models.\n")
        else:
            f.write("⚠ Some tests failed. Recommendations:\n")
            f.write("  1. Review failed test output for details\n")
            f.write("  2. Check CUDA installation and compatibility\n")
            f.write("  3. Verify PyTorch CUDA support\n")
            f.write("  4. Run tests with --verbose flag for more details\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Human-readable summary saved to: {txt_report_file}")


def run_specific_test_suite(suite_name: str):
    """Run a specific test suite."""
    suites = {
        'core': ['test_cuda_graph_capture.py'],
        'bucket': ['test_bucket_management.py'],
        'validation': ['test_output_validation.py'],
        'benchmark': ['test_benchmarking.py'],
        'integration': ['test_integration.py', 'test_llama_integration.py'],
        'performance': ['test_performance.py'],
        'all': [
            'test_cuda_graph_capture.py',
            'test_bucket_management.py',
            'test_output_validation.py',
            'test_benchmarking.py',
            'test_integration.py',
            'test_performance.py',
            'test_llama_integration.py',
        ],
    }
    
    if suite_name not in suites:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {list(suites.keys())}")
        return
    
    print(f"Running test suite: {suite_name}")
    print()
    
    # Temporarily modify test_files
    global test_files
    original_test_files = test_files
    test_files = suites[suite_name]
    
    try:
        run_tests()
    finally:
        test_files = original_test_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CUDA Graph Compilation tests')
    parser.add_argument(
        '--suite',
        choices=['core', 'integration', 'all'],
        default='all',
        help='Test suite to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.verbose:
        os.environ['PYTEST_VERBOSE'] = '1'
    
    # Run tests
    run_specific_test_suite(args.suite)
    
    # Exit with appropriate code
    summary = run_tests()
    sys.exit(0 if summary['all_passed'] else 1)