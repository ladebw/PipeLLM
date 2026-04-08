"""
Tests for llama.cpp integration with CUDA graph compilation.

These tests verify the integration of CUDA graph capture with
llama.cpp-style transformer models.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from typing import List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cuda_graph.llama_integration import (
    LlamaLayerGraphAdapter,
    LlamaModelGraphWrapper,
    wrap_model_with_cuda_graphs
)
from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType


@pytest.fixture
def cuda_available():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return True


@pytest.fixture
def mock_transformer_layer():
    """Create a mock transformer layer for testing."""
    
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
            
            # Mock attention computation
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Mock attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Mock softmax
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Mock attention output
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            # Mock past key/value update
            if past_key_value is not None:
                past_key, past_value = past_key_value
                new_past_key = torch.cat([past_key, k], dim=2)
                new_past_value = torch.cat([past_value, v], dim=2)
                past_key_value = (new_past_key, new_past_value)
            else:
                past_key_value = (k, v)
            
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
    
    return MockTransformerLayer()


@pytest.fixture
def mock_llama_model(mock_transformer_layer):
    """Create a mock llama model for testing."""
    
    class MockLlamaConfig:
        def __init__(self):
            self.hidden_size = 4096
            self.num_attention_heads = 32
            self.num_hidden_layers = 4  # Small for testing
            self.vocab_size = 32000
    
    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockLlamaConfig()
            
            # Create layers
            self.layers = nn.ModuleList([
                mock_transformer_layer.__class__(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads
                )
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
            
            # Mock return dict
            return {
                'logits': logits,
                'past_key_values': next_decoder_cache if use_cache else None,
                'hidden_states': all_hidden_states,
                'attentions': all_attentions,
            }
    
    return MockLlamaModel()


class TestLlamaLayerGraphAdapter:
    """Test layer adapter functionality."""
    
    def test_initialization(self, cuda_available, mock_transformer_layer):
        """Test layer adapter initialization."""
        graph_manager = CUDAGraphManager()
        
        adapter = LlamaLayerGraphAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            graph_manager=graph_manager,
            hidden_size=4096,
            num_heads=32,
            head_dim=128
        )
        
        assert adapter.layer == mock_transformer_layer
        assert adapter.layer_idx == 0
        assert adapter.graph_manager == graph_manager
        assert adapter.hidden_size == 4096
        assert adapter.num_heads == 32
        assert adapter.head_dim == 128
        assert not adapter.graphs_captured
        assert adapter.captured_seq_len is None
    
    def test_forward_eager(self, cuda_available, mock_transformer_layer):
        """Test forward pass with eager execution."""
        graph_manager = CUDAGraphManager()
        
        adapter = LlamaLayerGraphAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            graph_manager=graph_manager,
        )
        
        # Create test inputs
        batch_size = 2
        seq_len = 128
        hidden_size = 4096
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass (should use eager since no graphs captured)
        outputs = adapter.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=False,
        )
        
        # Should return tuple of (hidden_states, past_key_value, attentions)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        
        hidden_states_out, past_key_value, attentions = outputs
        
        assert hidden_states_out.shape == (batch_size, seq_len, hidden_size)
        assert isinstance(past_key_value, tuple)
        assert len(past_key_value) == 2  # key, value
        assert attentions is None  # output_attentions=False
    
    def test_prepare_graph_inputs(self, cuda_available, mock_transformer_layer):
        """Test graph input preparation."""
        graph_manager = CUDAGraphManager()
        
        adapter = LlamaLayerGraphAdapter(
            layer=mock_transformer_layer,
            layer_idx=0,
            graph_manager=graph_manager,
            num_heads=32,
            head_dim=128
        )
        
        # Create test inputs
        batch_size = 2
        seq_len = 128
        hidden_size = 4096
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        
        # Test with all inputs provided
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        
        inputs = adapter._prepare_graph_inputs(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None
        )
        
        assert isinstance(inputs, list)
        assert len(inputs) == 4  # hidden_states, mask, position_ids, past_kv
        
        # Test with None inputs (should create defaults)
        inputs2 = adapter._prepare_graph_inputs(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None
        )
        
        assert len(inputs2) == 4
        assert inputs2[1] is not None  # attention_mask created
        assert inputs2[2] is not None  # position_ids created
        assert inputs2[3] is not None  # past_key_value created


class TestLlamaModelGraphWrapper:
    """Test model wrapper functionality."""
    
    def test_initialization(self, cuda_available, mock_llama_model):
        """Test model wrapper initialization."""
        model = mock_llama_model().to('cuda')
        
        wrapper = LlamaModelGraphWrapper(model)
        
        assert wrapper.model == model
        assert wrapper.device.type == 'cuda'
        assert wrapper.graph_manager is not None
        assert isinstance(wrapper.capture_seq_lens, list)
        assert len(wrapper.wrapped_layers) == model.config.num_hidden_layers
        
        # Check statistics
        assert wrapper.stats["total_tokens"] == 0
        assert wrapper.stats["graph_tokens"] == 0
        assert wrapper.stats["eager_tokens"] == 0
    
    def test_wrap_model_layers(self, cuda_available, mock_llama_model):
        """Test layer wrapping logic."""
        model = mock_llama_model().to('cuda')
        
        wrapper = LlamaModelGraphWrapper(model)
        
        # Should have wrapped all layers
        assert len(wrapper.wrapped_layers) == model.config.num_hidden_layers
        
        # Each wrapped layer should be a LlamaLayerGraphAdapter
        for adapter in wrapper.wrapped_layers:
            assert isinstance(adapter, LlamaLayerGraphAdapter)
        
        # Original layers should be replaced with wrapped layers
        assert model.layers is not None
        assert len(model.layers) == model.config.num_hidden_layers
    
    def test_forward_pass(self, cuda_available, mock_llama_model):
        """Test forward pass through wrapped model."""
        model = mock_llama_model().to('cuda')
        
        wrapper = LlamaModelGraphWrapper(model)
        
        # Create test inputs
        batch_size = 1
        seq_len = 64
        
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device='cuda')
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
        
        # Forward pass
        outputs = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        
        # Should return dictionary
        assert isinstance(outputs, dict)
        
        # Check outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, model.config.vocab_size)
        
        assert 'past_key_values' in outputs
        assert outputs['past_key_values'] is not None
        assert len(outputs['past_key_values']) == model.config.num_hidden_layers
        
        # Check statistics
        stats = wrapper.get_stats()
        assert stats["total_tokens"] == batch_size * seq_len
    
    def test_benchmark(self, cuda_available, mock_llama_model):
        """Test benchmarking functionality."""
        model = mock_llama_model().to('cuda')
        
        wrapper = LlamaModelGraphWrapper(model)
        
        # Run benchmark
        results = wrapper.benchmark(
            seq_len=128,
            num_iterations=5  # Small for quick test
        )
        
        # Check results structure
        assert "seq_len" in results
        assert results["seq_len"] == 128
        
        assert "eager_avg_ms" in results
        assert results["eager_avg_ms"] > 0
        
        assert "num_layers" in results
        assert results["num_layers"] == model.config.num_hidden_layers
        
        assert "device" in results
        assert "cuda" in results["device"]
        
        assert "graphs_captured" in results
        # No graphs captured yet, so should be 0
        assert results["graphs_captured"] == 0
    
    def test_get_stats(self, cuda_available, mock_llama_model):
        """Test statistics collection."""
        model = mock_llama_model().to('cuda')
        
        wrapper = LlamaModelGraphWrapper(model)
        
        # Do some forward passes
        batch_size = 1
        seq_len = 64
        
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device='cuda')
        
        for _ in range(3):
            wrapper.forward(input_ids=input_ids)
        
        # Get statistics
        stats = wrapper.get_stats()
        
        # Check combined stats
        assert "total_tokens" in stats
        assert stats["total_tokens"] == 3 * batch_size * seq_len
        
        assert "graph_executions" in stats
        assert "eager_executions" in stats
        assert "graph_hit_rate" in stats
        
        # Should have token hit rate
        if stats["total_tokens"] > 0:
            assert "token_hit_rate" in stats
            assert 0 <= stats["token_hit_rate"] <= 1


class TestWrapModelFunction:
    """Test convenience wrapper function."""
    
    def test_wrap_model(self, cuda_available, mock_llama_model):
        """Test wrap_model_with_cuda_graphs function."""
        model = mock_llama_model().to('cuda')
        
        # Wrap model
        wrapped_model = wrap_model_with_cuda_graphs(
            model=model,
            capture_seq_lens=[512, 1024]
        )
        
        # Should return LlamaModelGraphWrapper
        assert isinstance(wrapped_model, LlamaModelGraphWrapper)
        
        # Check configuration
        assert wrapped_model.capture_seq_lens == [512, 1024]
        assert len(wrapped_model.wrapped_layers) == model.config.num_hidden_layers
        
        # Model should still work
        input_ids = torch.randint(0, 32000, (1, 64), device='cuda')
        outputs = wrapped_model.forward(input_ids=input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (1, 64, model.config.vocab_size)
    
    def test_wrap_model_defaults(self, cuda_available, mock_llama_model):
        """Test wrap_model_with_cuda_graphs with default parameters."""
        model = mock_llama_model().to('cuda')
        
        # Wrap with defaults
        wrapped_model = wrap_model_with_cuda_graphs(model=model)
        
        assert isinstance(wrapped_model, LlamaModelGraphWrapper)
        
        # Should have default capture sequence lengths
        default_lengths = [512, 1024, 2048, 4096]
        assert wrapped_model.capture_seq_lens == default_lengths


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])