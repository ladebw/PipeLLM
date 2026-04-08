#!/usr/bin/env python3
"""
Verify that all test files are properly structured and can be imported.
"""

import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'cuda_graph'))

def verify_test_file(test_file: str) -> bool:
    """Verify that a test file can be imported and has basic structure."""
    try:
        module = __import__(test_file)
        print(f"[OK] {test_file}.py imports successfully")
        
        # Check for common test patterns
        module_dict = module.__dict__
        
        # Check for test classes
        test_classes = [name for name in module_dict if name.startswith('Test')]
        if test_classes:
            print(f"  Found test classes: {', '.join(test_classes)}")
        else:
            print(f"  Warning: No test classes found starting with 'Test'")
        
        # Check for pytest fixtures
        fixtures = [name for name in module_dict if 'fixture' in str(module_dict.get(name, '')).lower()]
        if fixtures:
            print(f"  Found fixtures: {', '.join(fixtures[:3])}{'...' if len(fixtures) > 3 else ''}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] {test_file}.py import failed: {e}")
        return False

def main():
    """Main verification function."""
    print("Verifying test file structure...")
    print("=" * 60)
    
    test_files = [
        'test_cuda_graph_capture',
        'test_bucket_management',
        'test_output_validation',
        'test_llama_integration',
        'test_benchmarking',
        'test_integration',
        'test_performance',
        'test_utils',
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        print(f"\nVerifying {test_file}.py:")
        print("-" * 40)
        
        if verify_test_file(test_file):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Verification complete: {success_count}/{total_count} files pass")
    
    if success_count == total_count:
        print("[SUCCESS] All test files verified successfully!")
        return 0
    else:
        print(f"[WARNING] {total_count - success_count} files need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())