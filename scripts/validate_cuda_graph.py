#!/usr/bin/env python3
"""
Standalone validation script for CUDA Graph Compilation (Phase 1, Task 1.5).

This script validates that CUDA graph execution produces identical outputs
to eager (baseline) execution for LLM decode loops.

Usage:
    python validate_cuda_graph.py [options]

Examples:
    # Quick validation with default settings
    python validate_cuda_graph.py
    
    # Thorough validation with specific context lengths
    python validate_cuda_graph.py --context-lengths 512 1024 2048 --iterations 5
    
    # Save results to directory
    python validate_cuda_graph.py --output-dir ./validation_results
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_graph.output_validation import (
    OutputValidator,
    create_default_validator
)
from cuda_graph.cuda_graph_capture import (
    CUDAGraphManager,
    GraphType,
    create_context_length_buckets
)
from cuda_graph.llama_integration import wrap_model_with_cuda_graphs


def create_mock_model_for_validation(hidden_size=4096, num_layers=4):
    """
    Create a mock model for validation testing.
    
    This creates a simple transformer-like model that can be used
    to validate CUDA graph functionality without requiring a full
    llama.cpp model.
    """
    
    class MockAttention(nn.Module):
        def __init__(self, hidden_size=4096, num_heads=32):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            
            # Mock projections
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, hidden_states, attention_mask=None, past_key_value=None):
            batch_size, seq_len, _ = hidden_states.shape
            
            # Simple attention computation
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # Mock attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = self.o_proj(attn_output)
            
            # Mock past key/value
            if past_key_value is not None:
                past_key, past_value = past_key_value
                new_past_key = torch.cat([past_key, k.unsqueeze(2)], dim=2)
                new_past_value = torch.cat([past_value, v.unsqueeze(2)], dim=2)
                past_key_value = (new_past_key, new_past_value)
            else:
                past_key_value = (k.unsqueeze(2), v.unsqueeze(2))
            
            return attn_output, past_key_value
    
    class MockMLP(nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
            self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
            self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        
        def forward(self, x):
            gate = torch.sigmoid(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = torch.relu(gate * up)
            return self.down_proj(hidden)
    
    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size=4096, num_heads=32):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(hidden_size)
            self.self_attn = MockAttention(hidden_size, num_heads)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size)
            self.mlp = MockMLP(hidden_size)
        
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
        ):
            # Self attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            
            hidden_states, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
            )
            hidden_states = residual + hidden_states
            
            # MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            outputs = (hidden_states,)
            
            if use_cache:
                outputs = outputs + (present_key_value,)
            
            if output_attentions:
                outputs = outputs + (None,)  # Mock attention weights
            
            return outputs
    
    class MockLlamaConfig:
        def __init__(self, hidden_size=4096, num_layers=4):
            self.hidden_size = hidden_size
            self.num_attention_heads = 32
            self.num_hidden_layers = num_layers
            self.vocab_size = 32000
    
    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=4096, num_layers=4):
            super().__init__()
            self.config = MockLlamaConfig(hidden_size, num_layers)
            
            # Create layers
            self.layers = nn.ModuleList([
                MockTransformerLayer(hidden_size, self.config.num_attention_heads)
                for _ in range(self.config.num_hidden_layers)
            ])
            
            # Mock embeddings
            self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.norm = nn.LayerNorm(self.config.hidden_size)
            
            # Mock LM head
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            batch_size, seq_len = input_ids.shape
            
            # Get embeddings
            hidden_states = self.embed_tokens(input_ids)
            
            # Prepare past key values
            if past_key_values is None:
                past_key_values = [None] * len(self.layers)
            
            # Process through layers
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            next_decoder_cache = () if use_cache else None
            
            for idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[idx],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[1],)
                
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[2],)
            
            # Final normalization
            hidden_states = self.norm(hidden_states)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # LM head
            logits = self.lm_head(hidden_states)
            
            if not return_dict:
                return tuple(
                    v for v in [logits, next_decoder_cache, all_hidden_states, all_attentions]
                    if v is not None
                )
            
            return {
                'logits': logits,
                'past_key_values': next_decoder_cache if use_cache else None,
                'hidden_states': all_hidden_states,
                'attentions': all_attentions,
            }
    
    return MockLlamaModel(hidden_size, num_layers)


def validate_simple_computation(args):
    """Validate simple computation with CUDA graphs."""
    print("=" * 80)
    print("Validating Simple Computation")
    print("=" * 80)
    
    # Create validator
    validator = create_default_validator()
    
    # Create a simple computation function
    def simple_computation(x, mask=None, pos=None, past_kv=None):
        # Simple deterministic computation
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        return output
    
    # Create graph manager
    graph_manager = CUDAGraphManager()
    
    # Capture graphs for different context lengths
    buckets = create_context_length_buckets()
    captured_graphs = {}
    
    for graph_type, seq_len in buckets.items():
        if seq_len in args.context_lengths:
            print(f"Capturing graph for {graph_type.name} (seq_len={seq_len})...")
            
            # Create sample inputs for capture
            batch_size = 1
            hidden_size = 4096
            
            sample_x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
            sample_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
            sample_pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            
            sample_inputs = [sample_x, sample_mask, sample_pos, None]
            
            # Capture graph
            graph_instance = graph_manager.capture_graph(
                graph_type=graph_type,
                capture_func=simple_computation
            )
            
            if graph_instance:
                captured_graphs[graph_type] = graph_instance
                print(f"  ✓ Graph captured successfully")
            else:
                print(f"  ✗ Failed to capture graph")
    
    # Validate across context lengths
    print("\n" + "=" * 80)
    print("Running Validation Tests")
    print("=" * 80)
    
    validation_results = {}
    
    for seq_len in args.context_lengths:
        print(f"\nValidating context length: {seq_len}")
        
        # Find best graph type
        graph_type = graph_manager.get_best_fit_graph(seq_len)
        
        if graph_type and graph_type in captured_graphs:
            print(f"  Using graph type: {graph_type.name}")
            
            # Create test inputs
            batch_size = args.batch_size
            hidden_size = 4096
            
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            
            x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
            mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
            pos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            
            inputs = [x, mask, pos, None]
            
            # Define eager function
            def eager_func(*args):
                return simple_computation(*args)
            
            # Define graph function
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
            
            validation_results[seq_len] = result
            
            # Print result
            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status}")
            print(f"  Max Absolute Error: {result.max_absolute_error:.2e}")
            print(f"  Max Relative Error: {result.max_relative_error:.2e}")
            
            if not result.passed and result.error_details:
                print(f"  Errors detected")
        else:
            print(f"  ✗ No suitable graph available for seq_len={seq_len}")
            # Create a failed result
            validation_results[seq_len] = validator.validate_single_forward(
                eager_func=simple_computation,
                graph_func=simple_computation,  # Same function
                inputs=[torch.randn(1, seq_len, 4096, device='cuda')],
                sequence_length=seq_len,
                batch_size=1,
                graph_type=None
            )
            validation_results[seq_len].passed = False
            validation_results[seq_len].warnings.append("No graph available for validation")
    
    return validation_results


def validate_model_wrapper(args):
    """Validate model wrapper with CUDA graphs."""
    print("=" * 80)
    print("Validating Model Wrapper")
    print("=" * 80)
    
    # Create validator
    validator = create_default_validator()
    
    # Create mock model
    print("Creating mock model for validation...")
    model = create_mock_model_for_validation(
        hidden_size=4096,
        num_layers=args.num_layers
    ).to('cuda')
    
    # Wrap model with CUDA graphs
    print("Wrapping model with CUDA graphs...")
    wrapped_model = wrap_model_with_cuda_graphs(
        model=model,
        capture_seq_lens=args.context_lengths
    )
    
    # Pre-capture graphs
    print("Pre-capturing graphs...")
    wrapped_model.capture_all_graphs()
    
    # Validate
    print("\nRunning validation tests...")
    validation_results = validator.validate_model_wrapper(
        model_wrapper=wrapped_model,
        context_lengths=args.context_lengths,
        batch_size=args.batch_size,
        vocab_size=32000,
        num_iterations=args.iterations
    )
    
    # Print results
    for seq_len, result in validation_results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"\nContext Length: {seq_len} [{status}]")
        print(f"  Max Absolute Error: {result.max_absolute_error:.2e}")
        print(f"  Max Relative Error: {result.max_relative_error:.2e}")
        print(f"  Tensors Compared: {result.num_tensors_compared}")
        
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate CUDA Graph Output Correctness (Phase 1, Task 1.5)"
    )
    
    parser.add_argument(
        '--mode',
        choices=['simple', 'model', 'both'],
        default='both',
        help='Validation mode: simple computation or full model wrapper'
    )
    
    parser.add_argument(
        '--context-lengths',
        type=int,
        nargs='+',
        default=[512, 1024, 2048, 4096],
        help='Context lengths to validate (default: 512 1024 2048 4096)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for validation (default: 1)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations per context length (default: 3)'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=4,
        help='Number of layers in mock model (default: 4)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save validation results'
    )
    
    parser.add_argument(
        '--tolerance-abs',
        type=float,
        default=1e-5,
        help='Absolute tolerance for validation (default: 1e-5)'
    )
    
    parser.add_argument(
        '--tolerance-rel',
        type=float,
        default=1e-4,
        help='Relative tolerance for validation (default: 1e-4)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Validation requires CUDA.")
        sys.exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    all_results = {}
    
    # Run validation based on mode
    if args.mode in ['simple', 'both']:
        simple_results = validate_simple_computation(args)
        all_results['simple'] = simple_results
    
    if args.mode in ['model', 'both']:
        model_results = validate_model_wrapper(args)
        all_results['model'] = model_results
    
    # Create validator for report generation
    validator = OutputValidator(
        absolute_tolerance=args.tolerance_abs,
        relative_tolerance=args.tolerance_rel
    )
    
    # Generate and print summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    for mode, results in all_results.items():
        print(f"\n{mode.upper()} VALIDATION:")
        print("-" * 40)
        
        mode_tests = len(results)
        mode_passed = sum(1 for r in results.values() if r.passed)
        
        total_tests += mode_tests
        passed_tests += mode_passed
        
        print(f"  Tests: {mode_tests}")
        print(f"  Passed: {mode_passed}")
        print(f"  Failed: {mode_tests - mode_passed}")
        print(f"  Pass Rate: {mode_passed/mode_tests*100:.1f}%")
        
        # Check for any failures
        failed_lengths = [seq_len for seq_len, result in results.items() if not result.passed]
        if failed_lengths:
            print(f"  Failed Context Lengths: {failed_lengths}")
    
    print("\n" + "=" * 80)
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 80)
    
    # Save results if requested
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each mode's results
        for mode, results in all_results.items():
            mode_dir = args.output_dir / mode
            validator.save_validation_results(
                validation_results=results,
                output_dir=mode_dir,
                filename=f"{mode}_validation_results.json"
            )
        
        print(f"\nResults saved to: {args.output_dir}")
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        print("\n✓ All validation tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total_tests - passed_tests} validation test(s) failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()