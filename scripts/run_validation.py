#!/usr/bin/env python3
"""
Quick validation script for CUDA Graph Output Correctness.

This is a simplified version of the validation script that can be run
as part of CI/CD or quick verification.
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_graph.output_validation import OutputValidator
from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType


def quick_validation():
    """Run a quick validation test."""
    print("Running Quick CUDA Graph Validation...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Skipping validation.")
        return False
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Create validator
    validator = OutputValidator(
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4
    )
    
    # Simple computation function
    def simple_computation(x, mask=None, pos=None, past_kv=None):
        # Deterministic computation
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        return output
    
    # Create graph manager
    graph_manager = CUDAGraphManager()
    
    # Test context lengths
    test_lengths = [512, 1024]
    all_passed = True
    
    for seq_len in test_lengths:
        print(f"\nTesting seq_len={seq_len}...")
        
        # Determine graph type
        if seq_len <= 512:
            graph_type = GraphType.SHORT
        elif seq_len <= 1024:
            graph_type = GraphType.STANDARD
        else:
            graph_type = GraphType.MEDIUM
        
        # Capture graph
        print(f"  Capturing {graph_type.name} graph...")
        graph_instance = graph_manager.capture_graph(
            graph_type=graph_type,
            capture_func=simple_computation
        )
        
        if not graph_instance:
            print(f"  ✗ Failed to capture graph")
            all_passed = False
            continue
        
        # Create test inputs
        batch_size = 1
        hidden_size = 4096
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = [x, mask, pos, None]
        
        # Define functions
        def eager_func(*args):
            return simple_computation(*args)
        
        def graph_func(*args):
            return graph_manager.execute_graph(graph_type, args)
        
        # Run validation
        result = validator.validate_single_forward(
            eager_func=eager_func,
            graph_func=graph_func,
            inputs=inputs,
            sequence_length=seq_len,
            batch_size=batch_size,
            graph_type=graph_type.name
        )
        
        # Check result
        if result.passed:
            print(f"  ✓ PASSED (max error: {result.max_absolute_error:.2e})")
        else:
            print(f"  ✗ FAILED (max error: {result.max_absolute_error:.2e})")
            all_passed = False
    
    return all_passed


def main():
    """Main entry point."""
    try:
        success = quick_validation()
        
        print("\n" + "=" * 60)
        if success:
            print("VALIDATION PASSED")
            print("CUDA graph execution produces identical outputs to eager execution.")
            return 0
        else:
            print("VALIDATION FAILED")
            print("CUDA graph execution does not match eager execution.")
            return 1
            
    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())