#!/usr/bin/env python3
"""
Test imports for the test files.
"""

import sys
import os

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'cuda_graph'))

print("Testing imports...")
print("=" * 60)

# Test 1: Import benchmarks module
print("\n1. Testing benchmarks module import:")
try:
    import benchmarks
    print("   [OK] benchmarks module imported")
    
    # Try to import from it
    from benchmarks import cuda_graph_benchmark
    print("   [OK] cuda_graph_benchmark imported")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 2: Import test_benchmarking
print("\n2. Testing test_benchmarking import:")
try:
    import test_benchmarking
    print("   [OK] test_benchmarking imported")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test 3: Import test_integration
print("\n3. Testing test_integration import:")
try:
    import test_integration
    print("   [OK] test_integration imported")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n" + "=" * 60)
print("Import test complete")