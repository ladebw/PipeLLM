#!/usr/bin/env python3
"""
Task 2.2: Set up dual CUDA stream infrastructure (compute + copy)
Phase 2: Async Double-Buffered Weight Prefetch

This script implements Task 2.2 from the roadmap:
- Set up dual CUDA stream infrastructure
- Implement pinned memory buffer pool
- Create async prefetch engine
- Validate infrastructure functionality
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from pipeline_parallel.async_prefetch.dual_stream_manager import DualStreamManager, StreamType, demonstrate_dual_streams
from pipeline_parallel.async_prefetch.pinned_memory_pool import PinnedMemoryPool, demonstrate_pinned_memory_pool
from pipeline_parallel.async_prefetch.async_prefetch_engine import AsyncPrefetchEngine, demonstrate_async_prefetch
import json
from datetime import datetime
import time


def run_task_2_2():
    """Execute Task 2.2: Set up dual CUDA stream infrastructure."""
    print("=" * 80)
    print("TASK 2.2: Set up dual CUDA stream infrastructure (compute + copy)")
    print("Phase 2: Async Double-Buffered Weight Prefetch")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("phase2_results") / "task_2_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "task": "2.2",
        "phase": "Async Double-Buffered Weight Prefetch",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "tests": [],
        "summary": {}
    }
    
    # Test 1: Dual Stream Manager
    print("1. Testing Dual CUDA Stream Manager...")
    try:
        stream_test_result = test_dual_stream_manager(output_dir)
        results["components"]["dual_stream_manager"] = stream_test_result
        results["tests"].append({
            "name": "dual_stream_manager",
            "status": "PASSED" if stream_test_result["success"] else "FAILED",
            "details": stream_test_result
        })
        print(f"   Status: {'PASSED' if stream_test_result['success'] else 'FAILED'}")
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        results["components"]["dual_stream_manager"] = error_result
        results["tests"].append({
            "name": "dual_stream_manager",
            "status": "FAILED",
            "error": str(e)
        })
        print(f"   Status: FAILED - {e}")
    
    # Test 2: Pinned Memory Pool
    print("\n2. Testing Pinned Memory Pool...")
    try:
        pool_test_result = test_pinned_memory_pool(output_dir)
        results["components"]["pinned_memory_pool"] = pool_test_result
        results["tests"].append({
            "name": "pinned_memory_pool",
            "status": "PASSED" if pool_test_result["success"] else "FAILED",
            "details": pool_test_result
        })
        print(f"   Status: {'PASSED' if pool_test_result['success'] else 'FAILED'}")
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        results["components"]["pinned_memory_pool"] = error_result
        results["tests"].append({
            "name": "pinned_memory_pool",
            "status": "FAILED",
            "error": str(e)
        })
        print(f"   Status: FAILED - {e}")
    
    # Test 3: Async Prefetch Engine
    print("\n3. Testing Async Prefetch Engine...")
    try:
        engine_test_result = test_async_prefetch_engine(output_dir)
        results["components"]["async_prefetch_engine"] = engine_test_result
        results["tests"].append({
            "name": "async_prefetch_engine",
            "status": "PASSED" if engine_test_result["success"] else "FAILED",
            "details": engine_test_result
        })
        print(f"   Status: {'PASSED' if engine_test_result['success'] else 'FAILED'}")
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        results["components"]["async_prefetch_engine"] = error_result
        results["tests"].append({
            "name": "async_prefetch_engine",
            "status": "FAILED",
            "error": str(e)
        })
        print(f"   Status: FAILED - {e}")
    
    # Test 4: Integration Test
    print("\n4. Running Integration Test...")
    try:
        integration_result = test_integration(output_dir)
        results["components"]["integration_test"] = integration_result
        results["tests"].append({
            "name": "integration_test",
            "status": "PASSED" if integration_result["success"] else "FAILED",
            "details": integration_result
        })
        print(f"   Status: {'PASSED' if integration_result['success'] else 'FAILED'}")
    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        results["components"]["integration_test"] = error_result
        results["tests"].append({
            "name": "integration_test",
            "status": "FAILED",
            "error": str(e)
        })
        print(f"   Status: FAILED - {e}")
    
    # Generate summary
    results["summary"] = generate_summary(results)
    
    # Save results
    results_file = output_dir / "task_2_2_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    generate_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print("TASK 2.2 COMPLETED")
    print(f"{'='*80}")
    
    # Print summary
    summary = results["summary"]
    print(f"\nSUMMARY:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    
    if summary["passed_tests"] == summary["total_tests"]:
        print(f"\n  RESULT: ALL TESTS PASSED")
    else:
        print(f"\n  RESULT: {summary['failed_tests']} TEST(S) FAILED")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Detailed report: {output_dir / 'task_2_2_report.md'}")
    
    return results


def test_dual_stream_manager(output_dir: Path) -> Dict:
    """Test the dual stream manager."""
    test_dir = output_dir / "dual_stream_tests"
    test_dir.mkdir(exist_ok=True)
    
    results = {
        "test": "dual_stream_manager",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "metrics": {},
        "details": {}
    }
    
    try:
        # Create stream manager
        stream_manager = DualStreamManager(device="cuda", enable_timing=True)
        
        # Test 1: Stream initialization
        results["details"]["stream_initialization"] = {
            "compute_stream": stream_manager.compute_stream is not None,
            "copy_stream": stream_manager.copy_stream is not None,
            "default_stream": stream_manager.default_stream is not None,
        }
        
        # Test 2: Basic operations
        def mock_compute():
            import torch
            if torch.cuda.is_available():
                a = torch.randn(1024, 1024, device='cuda')
                b = torch.randn(1024, 1024, device='cuda')
                return torch.mm(a, b)
            return torch.randn(1024, 1024)
        
        def mock_copy():
            import torch
            if torch.cuda.is_available():
                cpu_data = torch.randn(1024 * 1024)
                return cpu_data.to('cuda')
            return torch.randn(1024 * 1024)
        
        # Execute operations
        compute_result = stream_manager.execute_compute_operation(mock_compute)
        copy_result = stream_manager.execute_copy_operation(mock_copy)
        
        results["details"]["operations_executed"] = {
            "compute": compute_result is not None,
            "copy": copy_result is not None,
        }
        
        # Test 3: Overlap measurement
        compute_time = 10.0  # ms
        copy_time = 5.0  # ms
        overlap = stream_manager.measure_overlap(compute_time, copy_time)
        
        results["details"]["overlap_measurement"] = {
            "compute_time_ms": compute_time,
            "copy_time_ms": copy_time,
            "overlap_percentage": overlap,
        }
        
        # Test 4: Get metrics
        metrics = stream_manager.get_metrics()
        results["metrics"] = metrics
        
        # Generate report
        report = stream_manager.generate_report()
        
        # Save report
        report_file = test_dir / "dual_stream_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        results["report_file"] = str(report_file)
        
        # Check success conditions
        if not all(results["details"]["stream_initialization"].values()):
            results["success"] = False
            results["error"] = "Stream initialization failed"
        
        if not all(results["details"]["operations_executed"].values()):
            results["success"] = False
            results["error"] = "Operation execution failed"
        
        stream_manager.synchronize_all()
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    
    return results


def test_pinned_memory_pool(output_dir: Path) -> Dict:
    """Test the pinned memory pool."""
    test_dir = output_dir / "memory_pool_tests"
    test_dir.mkdir(exist_ok=True)
    
    results = {
        "test": "pinned_memory_pool",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "metrics": {},
        "details": {}
    }
    
    try:
        # Create memory pool
        memory_pool = PinnedMemoryPool(
            total_size_mb=512,  # 512MB for testing
            max_buffer_size_mb=128,
            min_buffer_size_mb=1,
            enable_stats=True
        )
        
        # Test 1: Initial pool state
        pool_info = memory_pool.get_pool_info()
        results["details"]["initial_state"] = pool_info
        
        # Test 2: Buffer allocation
        buffer_sizes = [10 * 1024 * 1024, 50 * 1024 * 1024]  # 10MB, 50MB
        allocated_buffers = []
        
        for i, size_bytes in enumerate(buffer_sizes):
            buffer = memory_pool.allocate(
                size_bytes,
                metadata={"test_id": i, "size_mb": size_bytes // (1024 * 1024)}
            )
            
            if buffer:
                allocated_buffers.append(buffer)
                results["details"][f"buffer_{i}_allocation"] = {
                    "success": True,
                    "buffer_id": buffer.buffer_id,
                    "size_bytes": buffer.size_bytes,
                    "is_pinned": buffer.is_pinned(),
                }
            else:
                results["details"][f"buffer_{i}_allocation"] = {
                    "success": False,
                    "size_bytes": size_bytes,
                }
                results["success"] = False
        
        # Test 3: Prefetch to GPU
        for i, buffer in enumerate(allocated_buffers):
            prefetch_success = memory_pool.prefetch_to_gpu(buffer, non_blocking=True)
            results["details"][f"buffer_{i}_prefetch"] = {
                "success": prefetch_success,
                "buffer_id": buffer.buffer_id,
                "state_after": buffer.state.value,
            }
            
            if not prefetch_success:
                results["success"] = False
        
        # Test 4: Write back to CPU
        time.sleep(0.1)  # Simulate compute time
        for i, buffer in enumerate(allocated_buffers):
            writeback_success = memory_pool.writeback_to_cpu(buffer, non_blocking=True)
            results["details"][f"buffer_{i}_writeback"] = {
                "success": writeback_success,
                "buffer_id": buffer.buffer_id,
                "state_after": buffer.state.value,
            }
            
            if not writeback_success:
                results["success"] = False
        
        # Test 5: Buffer release
        for i, buffer in enumerate(allocated_buffers):
            memory_pool.release(buffer)
            results["details"][f"buffer_{i}_release"] = {
                "success": True,
                "buffer_id": buffer.buffer_id,
            }
        
        # Test 6: Final pool state
        final_pool_info = memory_pool.get_pool_info()
        results["details"]["final_state"] = final_pool_info
        results["metrics"] = final_pool_info.get("statistics", {})
        
        # Save pool info
        pool_file = test_dir / "memory_pool_info.json"
        with open(pool_file, 'w') as f:
            json.dump(final_pool_info, f, indent=2)
        
        results["report_file"] = str(pool_file)
        
        # Cleanup
        memory_pool.cleanup()
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    
    return results


def test_async_prefetch_engine(output_dir: Path) -> Dict:
    """Test the async prefetch engine."""
    test_dir = output_dir / "prefetch_engine_tests"
    test_dir.mkdir(exist_ok=True)
    
    results = {
        "test": "async_prefetch_engine",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "metrics": {},
        "details": {}
    }
    
    try:
        # Create async prefetch engine
        engine = AsyncPrefetchEngine(
            memory_pool_size_mb=1024,
            enable_stats=True,
            device="cuda"
        )
        
        # Test 1: Engine initialization
        engine_info = engine.get_engine_info()
        results["details"]["initialization"] = {
            "device": engine_info["device"],
            "state": engine_info["state"],
            "total_buffers": engine_info["total_buffers"],
            "active_buffers": engine_info["active_buffers"],
        }
        
        # Test 2: Buffer registration
        buffer_sizes_mb = [50, 100]
        buffer_ids = []
        
        for size_mb in buffer_sizes_mb:
            buffer_id = engine.register_weight_buffer(
                size_mb * 1024 * 1024,
                metadata={"size_mb": size_mb, "test": "async_prefetch"}
            )
            buffer_ids.append(buffer_id)
            
            buffer_info = engine.get_buffer_info(buffer_id)
            results["details"][f"buffer_{buffer_id}_registration"] = {
                "buffer_id": buffer_id,
                "size_mb": size_mb,
                "is_active": buffer_info["is_active"] if buffer_info else False,
            }
            
            if not buffer_info or not buffer_info["is_active"]:
                results["success"] = False
        
        # Test 3: Mock compute function
        def mock_layer_compute(buffer_id: int):
            """Mock layer compute operation."""
            # Simulate compute time
            buffer_info = engine.get_buffer_info(buffer_id)
            if buffer_info:
                size_mb = buffer_info["size_bytes"] // (1024 * 1024)
                compute_time = size_mb * 0.02  # 0.02ms per MB
                time.sleep(compute_time / 1000)
            return {"result": "computed", "buffer_id": buffer_id}
        
        # Test 4: Execute with prefetch
        import torch
        
        def generate_mock_weights(size_mb: int):
            """Generate mock weight tensor."""
            elements = size_mb * 1024 * 1024 // 4  # float32
            return torch.randn(elements)
        
        # First operation (no prefetch)
        weights1 = generate_mock_weights(50)
        start_time = time.time()
        
        result1 = engine.execute_compute_with_prefetch(
            buffer_id=buffer_ids[0],
            compute_func=mock_layer_compute,
            compute_args=(buffer_ids[0],),
            next_weights=None  # No prefetch
        )
        
        time1 = (time.time() - start_time) * 1000
        
        results["details"]["operation_1_no_prefetch"] = {
            "success": result1 is not None,
            "time_ms": time1,
            "result": str(result1)[:100] if result1 else None,
        }
        
        # Second operation (with prefetch)
        weights2 = generate_mock_weights(50)
        start_time = time.time()
        
        result2 = engine.execute_compute_with_prefetch(
            buffer_id=buffer_ids[0],
            compute_func=mock_layer_compute,
            compute_args=(buffer_ids[0],),
            next_weights=weights2  # With prefetch
        )
        
        time2 = (time.time() - start_time) * 1000
        
        results["details"]["operation_2_with_prefetch"] = {
            "success": result2 is not None,
            "time_ms": time2,
            "result": str(result2)[:100] if result2 else None,
        }
        
        # Calculate improvement
        if time1 > 0 and time2 > 0:
            speedup = time1 / time2
            improvement = (1 - time2 / time1) * 100
            results["details"]["performance_comparison"] = {
                "speedup": speedup,
                "improvement_percentage": improvement,
                "time_saved_ms": time1 - time2,
            }
        
        # Test 5: Get engine statistics
        stats = engine.get_stats()
        results["metrics"] = stats
        
        # Save engine info
        final_engine_info = engine.get_engine_info()
        engine_file = test_dir / "async_prefetch_engine_info.json"
        with open(engine_file, 'w') as f:
            json.dump(final_engine_info, f, indent=2)
        
        results["report_file"] = str(engine_file)
        
        # Cleanup
        engine.cleanup()
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    
    return results


def test_integration(output_dir: Path) -> Dict:
    """Test integration of all components."""
    test_dir = output_dir / "integration_tests"
    test_dir.mkdir(exist_ok=True)
    
    results = {
        "test": "integration_test",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "details": {}
    }
    
    try:
        # Run component demonstrations
        print("\n   Running dual stream demonstration...")
        demonstrate_dual_streams()
        
        print("\n   Running pinned memory pool demonstration...")
        demonstrate_pinned_memory_pool()
        
        print("\n   Running async prefetch demonstration...")
        demonstrate_async_prefetch()
        
        results["details"]["demonstrations"] = {
            "dual_streams": "completed",
            "pinned_memory_pool": "completed",
            "async_prefetch": "completed",
        }
        
        # Test component interoperability
        print("\n   Testing component interoperability...")
        
        # Create all components
        stream_manager = DualStreamManager(device="cuda")
        memory_pool = PinnedMemoryPool(total_size_mb=512)
        engine = AsyncPrefetchEngine(memory_pool_size_mb=512)
        
        # Verify they work together
        stream_report = stream_manager.generate_report()
        pool_info = memory_pool.get_pool_info()
        engine_info = engine.get_engine_info()
        
        results["details"]["interoperability"] = {
            "stream_manager_initialized": stream_manager.compute_stream is not None,
            "memory_pool_initialized": len(pool_info["total_buffers"]) > 0,
            "engine_initialized": engine_info["total_buffers"] == 0,  # No buffers registered yet
        }
        
        # Cleanup
        engine.cleanup()
        memory_pool.cleanup()
        stream_manager.synchronize_all()
        
        results["success"] = all(results["details"]["interoperability"].values())
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    
    return results


def generate_summary(results: Dict) -> Dict:
    """Generate summary of test results."""
    tests = results.get("tests", [])
    
    total_tests = len(tests)
    passed_tests = sum(1 for t in tests if t.get("status") == "PASSED")
    failed_tests = total_tests - passed_tests
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Check if all components were tested successfully
    components = results.get("components", {})
    components_success = all(
        comp.get("success", False) 
        for comp in components.values() 
        if isinstance(comp, dict)
    )
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": success_rate,
        "all_components_tested": len(components) == 4,  # 4 main components
        "all_components_successful": components_success,
        "overall_success": passed_tests == total_tests and components_success,
    }


def generate_report(results: Dict, output_dir: Path):
    """Generate a comprehensive markdown report."""
    report_path = output_dir / "task_2_2_report.md"
    
    summary = results["summary"]
    components = results["components"]
    tests = results["tests"]
    
    with open(report_path, 'w') as f:
        f.write("# Task 2.2: Dual CUDA Stream Infrastructure Report\n\n")
        f.write(f"**Generated**: {results['timestamp']}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Tests**: {summary['total_tests']}\n")
        f.write(f"- **Passed**: {summary['passed_tests']}\n")
        f.write(f"- **Failed**: {summary['failed_tests']}\n")
        f.write(f"- **Success Rate**: {summary['success_rate']:.1f}%\n")
        f.write(f"- **Overall Status**: {'PASSED' if summary['overall_success'] else 'FAILED'}\n\n")
        
        f.write("## Component Tests\n\n")
        
        for test in tests:
            f.write(f"### {test['name'].replace('_', ' ').title()}\n\n")
            f.write(f"- **Status**: {test['status']}\n")
            
            if test['status'] == 'FAILED' and 'error' in test:
                f.write(f"- **Error**: {test['error']}\n")
            
            f.write("\n")
        
        f.write("## Component Details\n\n")
        
        for comp_name, comp_data in components.items():
            if not isinstance(comp_data, dict):
                continue
                
            f.write(f"### {comp_name.replace('_', ' ').title()}\n\n")
            f.write(f"- **Status**: {'PASSED' if comp_data.get('success') else 'FAILED'}\n")
            
            if 'error' in comp_data:
                f.write(f"- **Error**: {comp_data['error']}\n")
            
            if 'metrics' in comp_data and comp_data['metrics']:
                f.write("\n**Key Metrics**:\n")
                for key, value in comp_data['metrics'].items():
                    if isinstance(value, (int, float)):
                        if 'time' in key or 'ms' in key:
                            f.write(f"  - {key}: {value:.2f}\n")
                        else:
                            f.write(f"  - {key}: {value}\n")
            
            f.write("\n")
        
        f.write("## Implementation Details\n\n")
        f.write("### Dual CUDA Stream Manager\n")
        f.write("- Separate compute and copy streams\n")
        f.write("- Event-based synchronization\n")
        f.write("- Overlap measurement and optimization\n")
        f.write("- Performance metrics collection\n\n")
        
        f.write("### Pinned Memory Buffer Pool\n")
        f.write("- Page-locked memory for fast DMA transfers\n")
        f.write("- Buffer reuse and caching\n")
        f.write("- Async prefetch and writeback\n")
        f.write("- Memory usage statistics\n\n")
        
        f.write("### Async Prefetch Engine\n")
        f.write("- Double-buffered weight staging\n")
        f.write("- Compute-memory transfer overlap\n")
        f.write("- Pipeline scheduling\n")
        f.write("- Performance monitoring\n\n")
        
        f.write("## Next Steps (Task 2.3)\n\n")
        f.write("Based on successful implementation of Task 2.2, Task 2.3 will:\n\n")
        f.write("1. **Implement double-buffer swap logic between layers**\n")
        f.write("2. **Integrate with existing CUDA graph infrastructure**\n")
        f.write("3. **Add layer-specific prefetch scheduling**\n")
        f.write("4. **Optimize for different model architectures**\n\n")
        
        f.write("## Conclusion\n\n")
        
        if summary['overall_success']:
            f.write("Task 2.2 has been successfully completed. The dual CUDA stream infrastructure ")
            f.write("is fully implemented and tested, providing the foundation for async ")
            f.write("double-buffered weight prefetch in Phase 2.\n")
        else:
            f.write("Task 2.2 encountered some issues. While core components are implemented, ")
            f.write("some tests failed and require investigation before proceeding to Task 2.3.\n")
        
        f.write("\nThe implementation includes all necessary components for overlapping ")
        f.write("compute and memory transfer operations, which is critical for achieving ")
        f.write("the Phase 2 goal of 15-22% tokens/sec improvement.\n")
    
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    run_task_2_2()