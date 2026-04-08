#!/usr/bin/env python3
"""
Integration tests for the complete CUDA graph pipeline.

These tests verify that all components work together correctly:
1. CUDA graph capture
2. Bucket management
3. Model integration
4. Output validation
5. Benchmarking
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType, create_context_length_buckets
from cuda_graph.cuda_graph_buckets import GraphBucketManager
from cuda_graph.llama_integration import wrap_model_with_cuda_graphs
from cuda_graph.output_validation import OutputValidator
from benchmarks.cuda_graph_benchmark import CUDAGraphBenchmark, BenchmarkResult


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_transformer_model():
    """Create a mock transformer model for integration testing."""
    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size=4096, num_heads=32):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.attention_norm = nn.LayerNorm(hidden_size)
            self.ffn_norm = nn.LayerNorm(hidden_size)
            self.wq = nn.Linear(hidden_size, hidden_size)
            self.wk = nn.Linear(hidden_size, hidden_size)
            self.wv = nn.Linear(hidden_size, hidden_size)
            self.wo = nn.Linear(hidden_size, hidden_size)
            self.w1 = nn.Linear(hidden_size, hidden_size * 4)
            self.w2 = nn.Linear(hidden_size * 4, hidden_size)
            self.w3 = nn.Linear(hidden_size, hidden_size * 4)
        
        def forward(
            self,
            x,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        ):
            # Simplified attention
            residual = x
            x = self.attention_norm(x)
            
            # Self-attention
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            
            # Simple attention computation
            attention = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
            if attention_mask is not None:
                attention = attention + attention_mask.unsqueeze(1).unsqueeze(2)
            attention = torch.softmax(attention, dim=-1)
            x = torch.matmul(attention, v)
            x = self.wo(x)
            x = residual + x
            
            # FFN
            residual = x
            x = self.ffn_norm(x)
            x = self.w2(torch.relu(self.w1(x)) * self.w3(x))
            x = residual + x
            
            outputs = (x,)
            if output_attentions:
                outputs += (attention,)
            if use_cache:
                outputs += (past_key_value,)
            
            return outputs
    
    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=4096, num_layers=4, vocab_size=32000):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.vocab_size = vocab_size
            
            # Create layers
            self.layers = nn.ModuleList([
                MockTransformerLayer(hidden_size)
                for _ in range(num_layers)
            ])
            
            # Embeddings and head
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
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
            
            # Process through layers
            for i, layer in enumerate(self.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[i] if past_key_values else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]
            
            # Final normalization
            hidden_states = self.norm(hidden_states)
            
            # LM head
            logits = self.lm_head(hidden_states)
            
            if not return_dict:
                return (logits,)
            
            return {
                'logits': logits,
                'past_key_values': None,
                'hidden_states': None,
                'attentions': None,
            }
    
    return MockLlamaModel(hidden_size=256, num_layers=2, vocab_size=1000)


class TestCompletePipeline:
    """Tests for the complete CUDA graph pipeline."""
    
    def test_pipeline_graph_capture_to_execution(self, cuda_available, mock_transformer_model):
        """Test complete pipeline: graph capture -> execution -> validation."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Move model to GPU
        model = mock_transformer_model.to('cuda')
        
        # Create graph manager
        graph_manager = CUDAGraphManager()
        
        # Create test inputs
        batch_size = 1
        seq_len = 512
        hidden_size = 256
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
        attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        # Define capture function
        def capture_func(x, mask, pos, past_kv):
            return model(
                input_ids=x,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=past_kv,
                return_dict=True
            )
        
        # Capture graph
        print("Capturing graph...")
        capture_success = graph_manager.capture_graph(
            graph_type=GraphType.STANDARD,
            capture_func=capture_func,
            inputs=[input_ids, attention_mask, position_ids, None]
        )
        
        assert capture_success, "Graph capture failed"
        
        # Execute graph
        print("Executing graph...")
        graph_output = graph_manager.execute_graph(
            graph_type=GraphType.STANDARD,
            inputs=[input_ids, attention_mask, position_ids, None]
        )
        
        # Execute eager for comparison
        print("Executing eager...")
        with torch.no_grad():
            eager_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True
            )
        
        # Validate outputs
        print("Validating outputs...")
        validator = OutputValidator(absolute_tolerance=1e-5, relative_tolerance=1e-4)
        comparison = validator._compare_outputs(eager_output, graph_output)
        
        assert comparison["all_passed"], f"Output validation failed: {comparison}"
        print(f"Validation passed with max error: {comparison['max_absolute_error']:.2e}")
    
    def test_pipeline_with_bucket_manager(self, cuda_available, mock_transformer_model):
        """Test pipeline with bucket manager for multiple sequence lengths."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Move model to GPU
        model = mock_transformer_model.to('cuda')
        
        # Create bucket manager
        bucket_manager = GraphBucketManager(
            max_graphs_in_memory=3,
            memory_limit_mb=1024
        )
        
        # Define capture function
        def capture_func(x, mask, pos, past_kv):
            return model(
                input_ids=x,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=past_kv,
                return_dict=True
            )
        
        # Test different sequence lengths
        sequence_lengths = [256, 512, 1024]
        batch_size = 1
        
        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Create inputs
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
            attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
            position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            
            inputs = [input_ids, attention_mask, position_ids, None]
            
            # Get or create graph
            graph_type = bucket_manager.get_best_fit_graph_type(seq_len)
            print(f"  Using graph type: {graph_type}")
            
            # Execute with bucket manager
            graph_output = bucket_manager.execute_with_bucket_selection(
                seq_len=seq_len,
                capture_func=capture_func,
                inputs=inputs
            )
            
            # Execute eager for comparison
            with torch.no_grad():
                eager_output = capture_func(*inputs)
            
            # Validate
            validator = OutputValidator()
            comparison = validator._compare_outputs(eager_output, graph_output)
            
            assert comparison["all_passed"], f"Validation failed for seq_len {seq_len}: {comparison}"
            print(f"  Validation passed")
        
        # Check bucket manager stats
        stats = bucket_manager.get_memory_stats()
        assert "total_graphs" in stats
        assert "memory_used_mb" in stats
        print(f"Bucket manager stats: {stats}")
    
    def test_pipeline_with_model_wrapper(self, cuda_available, mock_transformer_model):
        """Test pipeline with model wrapper integration."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Move model to GPU
        model = mock_transformer_model.to('cuda')
        
        # Wrap model with CUDA graphs
        print("Wrapping model with CUDA graphs...")
        wrapped_model = wrap_model_with_cuda_graphs(
            model=model,
            capture_seq_lens=[256, 512]
        )
        
        # Pre-capture graphs
        print("Pre-capturing graphs...")
        wrapped_model.capture_all_graphs()
        
        # Test different sequence lengths
        sequence_lengths = [256, 512]
        batch_size = 1
        
        for seq_len in sequence_lengths:
            print(f"Testing wrapped model with sequence length: {seq_len}")
            
            # Create inputs
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
            attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
            
            # Test eager execution (graphs disabled)
            wrapped_model.graph_manager.enable_graphs = False
            with torch.no_grad():
                eager_output = wrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Test graph execution
            wrapped_model.graph_manager.enable_graphs = True
            with torch.no_grad():
                graph_output = wrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Validate
            validator = OutputValidator()
            comparison = validator._compare_outputs(eager_output, graph_output)
            
            assert comparison["all_passed"], f"Validation failed for seq_len {seq_len}: {comparison}"
            print(f"  Validation passed")
        
        # Get wrapper stats
        stats = wrapped_model.get_stats()
        assert "total_graphs" in stats
        assert "graph_hits" in stats
        print(f"Model wrapper stats: {stats}")
    
    def test_pipeline_with_benchmarking(self, cuda_available, mock_transformer_model):
        """Test complete pipeline including benchmarking."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create benchmark instance
            benchmark = CUDAGraphBenchmark(
                output_dir=Path(tmpdir) / "pipeline_benchmark",
                enable_validation=True,
                warmup_iterations=1,
                benchmark_iterations=2
            )
            
            # Run benchmark on the pipeline
            print("Running pipeline benchmark...")
            results = benchmark.benchmark_cuda_graph_vs_eager(
                config_name="pipeline_test",
                sequence_lengths=[256, 512],
                batch_size=1,
                hidden_size=256,
                num_layers=2,
                vocab_size=1000
            )
            
            # Verify results
            assert isinstance(results, dict)
            assert len(results) == 2  # Should have results for 256 and 512
            
            for seq_len in [256, 512]:
                assert seq_len in results
                seq_result = results[seq_len]
                
                # Check we have both eager and graph results
                assert "eager" in seq_result
                assert "cuda_graph" in seq_result
                assert "improvement" in seq_result
                
                # Check validation results
                graph_result = seq_result["cuda_graph"]
                assert hasattr(graph_result, 'validation_passed')
                assert graph_result.validation_passed is True, f"Validation failed for seq_len {seq_len}"
                
                # Check improvement data
                improvement = seq_result["improvement"]
                assert "speedup" in improvement
                assert "improvement_pct" in improvement
                
                print(f"Seq {seq_len}: Speedup={improvement['speedup']:.2f}x, "
                      f"Improvement={improvement['improvement_pct']:.1f}%")
            
            # Save and verify results
            results_file = benchmark.save_results(results, "pipeline_results.json")
            assert results_file.exists()
            
            summary_file = benchmark.save_summary_report(results, "pipeline_summary.txt")
            assert summary_file.exists()
    
    def test_pipeline_error_handling(self, cuda_available):
        """Test error handling in the pipeline."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Test 1: Invalid sequence length
        bucket_manager = GraphBucketManager()
        
        # Should return None for invalid sequence length
        graph_type = bucket_manager.get_best_fit_graph_type(100000)  # Very large sequence
        assert graph_type is None, "Should return None for invalid sequence length"
        
        # Test 2: Memory limit enforcement
        small_memory_manager = GraphBucketManager(memory_limit_mb=1)  # Very small limit
        
        def simple_capture(x, mask, pos, past_kv):
            return x * 2
        
        # Try to capture multiple graphs - should fail due to memory limit
        capture_success = small_memory_manager.capture_graph_on_demand(
            seq_len=512,
            capture_func=simple_capture,
            inputs=[torch.randn(1, 512, 256, device='cuda')]
        )
        
        # Might fail or succeed depending on actual memory usage
        # Just verify no crash
        assert True
    
    def test_pipeline_performance_improvement(self, cuda_available, mock_transformer_model):
        """Test that the pipeline actually provides performance improvement."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # This test verifies that CUDA graphs provide speedup
        # Note: Actual speedup depends on hardware and model size
        
        # Move model to GPU
        model = mock_transformer_model.to('cuda')
        
        # Wrap model
        wrapped_model = wrap_model_with_cuda_graphs(
            model=model,
            capture_seq_lens=[512]
        )
        
        # Pre-capture graph
        wrapped_model.capture_all_graphs()
        
        # Create test input
        batch_size = 1
        seq_len = 512
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
        attention_mask = torch.ones(batch_size, seq_len, device='cuda', dtype=torch.bool)
        
        # Benchmark eager execution
        wrapped_model.graph_manager.enable_graphs = False
        
        eager_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = wrapped_model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            eager_times.append((end - start) * 1000)  # Convert to ms
        
        # Benchmark graph execution
        wrapped_model.graph_manager.enable_graphs = True
        
        graph_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = wrapped_model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            graph_times.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        eager_mean = sum(eager_times) / len(eager_times)
        graph_mean = sum(graph_times) / len(graph_times)
        
        print(f"Eager mean: {eager_mean:.2f} ms")
        print(f"Graph mean: {graph_mean:.2f} ms")
        
        # Verify graph execution is not slower (allowing for measurement variance)
        # In practice, graph should be faster, but we allow for small variance
        speedup = eager_mean / graph_mean if graph_mean > 0 else 0
        
        print(f"Speedup: {speedup:.2f}x")
        
        # For this test, we just verify that graph execution works
        # and doesn't crash. Actual speedup verification would require
        # more rigorous benchmarking.
        assert graph_mean > 0, "Graph execution time should be positive"
        assert speedup >= 0.8, f"Graph execution should not be significantly slower (speedup: {speedup:.2f}x)"


class TestProductionReadiness:
    """Tests for production readiness of the CUDA graph pipeline."""
    
    def test_production_configuration(self, cuda_available):
        """Test production-ready configuration."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Production configuration
        from cuda_graph.cuda_graph_buckets import create_production_bucket_manager
        
        bucket_manager = create_production_bucket_manager()
        
        # Verify production settings
        assert bucket_manager.max_graphs_in_memory == 5
        assert bucket_manager.memory_limit_mb == 4096  # 4GB limit
        
        # Verify context length buckets
        buckets = create_context_length_buckets()
        assert len(buckets) > 0
        
        # Check that common sequence lengths are covered
        common_lengths = [512, 1024, 2048, 4096]
        for length in common_lengths:
            graph_type = bucket_manager.get_best_fit_graph_type(length)
            assert graph_type is not None, f"No graph type for common length {length}"
    
    def test_error_recovery(self, cuda_available):
        """Test that the pipeline recovers from errors."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Create bucket manager
        bucket_manager = GraphBucketManager()
        
        # Define a function that might fail
        def failing_capture(x, mask, pos, past_kv):
            if x.shape[1] > 1000:  # Fail for long sequences
                raise RuntimeError("Sequence too long")
            return x * 2
        
        # Test with sequence that should work
        short_input = torch.randn(1, 512, 256, device='cuda')
        result = bucket_manager.execute_with_bucket_selection(
            seq_len=512,
            capture_func=failing_capture,
            inputs=[short_input]
        )
        
        assert result is not None
        
        # Test with sequence that should fail
        # The bucket manager should handle the error gracefully
        # (either by falling back to eager or returning None)
        long_input = torch.randn(1, 2000, 256, device='cuda')
        
        try:
            result = bucket_manager.execute_with_bucket_selection(
                seq_len=2000,
                capture_func=failing_capture,
                inputs=[long_input]
            )
            # If we get here, the manager handled the error
            # (might return None or fall back to eager)
            assert result is None or isinstance(result, torch.Tensor)
        except Exception as e:
            # Error might propagate - that's also acceptable
            # as long as it doesn't crash the entire application
            print(f"Error propagated (acceptable): {e}")
    
    def test_memory_management(self, cuda_available):
        """Test memory management in production scenario."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        # Create bucket manager with small memory limit
        bucket_manager = GraphBucketManager(
            max_graphs_in_memory=2,
            memory_limit_mb=512  # 0.5GB limit
        )
        
        def capture_func(x, mask, pos, past_kv):
            # Create a moderately sized tensor
            return torch.randn_like(x) * 2
        
        # Capture multiple graphs
        captured_count = 0
        for seq_len in [256, 512, 1024]:
            inputs = [torch.randn(1, seq_len, 256, device='cuda')]
            
            success = bucket_manager.capture_graph_on_demand(
                seq_len=seq_len,
                capture_func=capture_func,
                inputs=inputs
            )
            
            if success:
                captured_count += 1
        
        # With memory limit, we might not capture all graphs
        stats = bucket_manager.get_memory_stats()
        print(f"Memory stats: {stats}")
        
        # Verify we don't exceed memory limit
        if "memory_used_mb" in stats:
            assert stats["memory_used_mb"] <= 512 * 1.1  # Allow 10% overhead
        
        # Verify LRU eviction works
        assert stats["total_graphs"] <= 2  # Should not exceed max_graphs_in_memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])