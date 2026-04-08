"""
Integration with llama.cpp-style models for CUDA graph compilation.

This module provides adapters to integrate CUDA graph compilation
with models that follow the llama.cpp architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from .cuda_graph_capture import CUDAGraphManager, GraphType, benchmark_graph_vs_eager

logger = logging.getLogger(__name__)


class LlamaLayerGraphAdapter:
    """
    Adapter for llama.cpp-style transformer layers to work with CUDA graphs.
    
    This class wraps a transformer layer and manages CUDA graph capture
    and execution for the layer's forward pass.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        graph_manager: CUDAGraphManager,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128
    ):
        """
        Initialize the layer adapter.
        
        Args:
            layer: The transformer layer to wrap
            layer_idx: Index of this layer in the model
            graph_manager: CUDA graph manager instance
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
        """
        self.layer = layer
        self.layer_idx = layer_idx
        self.graph_manager = graph_manager
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Track whether graphs are captured for this layer
        self.graphs_captured = False
        self.captured_seq_len = None
        
        logger.info(f"Initialized graph adapter for layer {layer_idx}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with CUDA graph support.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_value: Previous key/value cache
            use_cache: Whether to return key/value cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (hidden_states, past_key_value, attentions)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Check if we have a suitable graph
        graph_type = self.graph_manager.get_best_fit_graph(seq_len)
        
        if graph_type is not None and self.graphs_captured:
            # Try to use graph execution
            try:
                # Prepare inputs for graph execution
                inputs = self._prepare_graph_inputs(
                    hidden_states, attention_mask, position_ids, past_key_value
                )
                
                # Execute with graph
                outputs = self.graph_manager.execute_graph(graph_type, inputs)
                
                # Unpack outputs
                hidden_states_out = outputs[0]
                past_key_value_out = outputs[1] if use_cache else None
                attentions_out = outputs[2] if output_attentions else None
                
                return hidden_states_out, past_key_value_out, attentions_out
                
            except Exception as e:
                logger.warning(f"Graph execution failed for layer {self.layer_idx}: {e}")
                # Fall through to eager execution
        
        # Eager execution fallback
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
    
    def capture_graphs(self, seq_len: int = 1024):
        """
        Capture CUDA graphs for this layer.
        
        Args:
            seq_len: Sequence length to capture graphs for
        """
        logger.info(f"Capturing graphs for layer {self.layer_idx} (seq_len={seq_len})")
        
        # Determine graph type
        graph_type = None
        for gt in GraphType:
            if gt.value >= seq_len:
                graph_type = gt
                break
        
        if graph_type is None:
            logger.warning(f"No suitable graph type for seq_len={seq_len}")
            return
        
        # Create capture function
        def capture_forward(*args):
            # Unpack inputs
            hidden_states, attention_mask, position_ids, past_key_value = args
            
            # Call layer forward
            outputs = self.layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False,
            )
            
            # Return as list for graph capture
            hidden_states_out, past_key_value_out, _ = outputs
            return [hidden_states_out, past_key_value_out, None]
        
        # Capture the graph
        try:
            self.graph_manager.capture_graph(
                graph_type=graph_type,
                capture_func=capture_forward,
                # Inputs will be created by capture_graph
            )
            self.graphs_captured = True
            self.captured_seq_len = seq_len
            logger.info(f"Successfully captured graphs for layer {self.layer_idx}")
        except Exception as e:
            logger.error(f"Failed to capture graphs for layer {self.layer_idx}: {e}")
            self.graphs_captured = False
    
    def _prepare_graph_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Prepare inputs for graph execution."""
        # Ensure all inputs are on the right device
        device = hidden_states.device
        
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=device
            )
        
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=device
            ).unsqueeze(0).expand(hidden_states.shape[0], -1)
        
        if past_key_value is None:
            # Create dummy past key/value
            batch_size = hidden_states.shape[0]
            past_key = torch.zeros(
                batch_size, self.num_heads, 0, self.head_dim,
                device=device
            )
            past_value = torch.zeros(
                batch_size, self.num_heads, 0, self.head_dim,
                device=device
            )
            past_key_value = (past_key, past_value)
        
        return [hidden_states, attention_mask, position_ids, past_key_value]


class LlamaModelGraphWrapper:
    """
    Wrapper for entire llama.cpp-style models with CUDA graph support.
    
    This class wraps a complete transformer model and manages
    CUDA graph capture and execution across all layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        graph_manager: Optional[CUDAGraphManager] = None,
        capture_seq_lens: List[int] = None
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model: The transformer model to wrap
            graph_manager: CUDA graph manager instance (creates new if None)
            capture_seq_lens: List of sequence lengths to capture graphs for
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Initialize graph manager
        if graph_manager is None:
            self.graph_manager = CUDAGraphManager(device=str(self.device))
        else:
            self.graph_manager = graph_manager
        
        # Default sequence lengths to capture
        if capture_seq_lens is None:
            self.capture_seq_lens = [512, 1024, 2048, 4096]
        else:
            self.capture_seq_lens = capture_seq_lens
        
        # Wrap model layers with graph adapters
        self.wrapped_layers = self._wrap_model_layers()
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "graph_tokens": 0,
            "eager_tokens": 0,
            "layer_stats": {},
        }
        
        logger.info(f"Initialized model wrapper with {len(self.wrapped_layers)} layers")
    
    def _wrap_model_layers(self) -> List[LlamaLayerGraphAdapter]:
        """Wrap all transformer layers with graph adapters."""
        wrapped_layers = []
        
        # Try to find the layers in the model
        # This depends on the specific model architecture
        
        if hasattr(self.model, 'layers'):
            # Standard llama.cpp architecture
            layers = self.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # HuggingFace Transformers architecture
            layers = self.model.model.layers
        else:
            # Try to find layers by name
            layers = []
            for name, module in self.model.named_modules():
                if 'layer' in name.lower() and isinstance(module, nn.ModuleList):
                    layers = module
                    break
        
        if not layers:
            logger.warning("Could not find transformer layers in model")
            return []
        
        # Get model config for layer parameters
        hidden_size = getattr(self.model.config, 'hidden_size', 4096)
        num_attention_heads = getattr(self.model.config, 'num_attention_heads', 32)
        head_dim = hidden_size // num_attention_heads
        
        # Wrap each layer
        for idx, layer in enumerate(layers):
            adapter = LlamaLayerGraphAdapter(
                layer=layer,
                layer_idx=idx,
                graph_manager=self.graph_manager,
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                head_dim=head_dim
            )
            wrapped_layers.append(adapter)
        
        # Replace original layers with wrapped layers
        if hasattr(self.model, 'layers'):
            self.model.layers = nn.ModuleList(wrapped_layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.model.model.layers = nn.ModuleList(wrapped_layers)
        
        return wrapped_layers
    
    def capture_all_graphs(self):
        """Capture CUDA graphs for all layers and sequence lengths."""
        logger.info(f"Capturing graphs for {len(self.capture_seq_lens)} sequence lengths")
        
        for seq_len in self.capture_seq_lens:
            logger.info(f"Capturing graphs for seq_len={seq_len}")
            
            for layer_adapter in self.wrapped_layers:
                layer_adapter.capture_graphs(seq_len=seq_len)
            
            # Synchronize after each sequence length
            torch.cuda.synchronize()
        
        logger.info("Graph capture complete")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        """
        Forward pass with CUDA graph support.
        
        This method intercepts the forward pass and uses graph execution
        where possible, falling back to eager execution otherwise.
        """
        batch_size, seq_len = input_ids.shape
        
        # Update statistics
        self.stats["total_tokens"] += batch_size * seq_len
        
        # Let the model handle the rest
        # The wrapped layers will use graph execution internally
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def benchmark(
        self,
        seq_len: int = 1024,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark graph vs eager execution for this model.
        
        Args:
            seq_len: Sequence length to benchmark
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking model with seq_len={seq_len}")
        
        # Create dummy inputs
        batch_size = 1
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
        
        # Create a simple forward function for benchmarking
        def eager_forward(*args):
            with torch.no_grad():
                return self.model(*args)
        
        # Run benchmark
        results = benchmark_graph_vs_eager(
            graph_manager=self.graph_manager,
            eager_func=eager_forward,
            seq_len=seq_len,
            num_iterations=num_iterations
        )
        
        # Add model-specific stats
        results.update({
            "num_layers": len(self.wrapped_layers),
            "device": str(self.device),
            "graphs_captured": len(self.graph_manager.graphs),
        })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        # Combine model stats with graph manager stats
        combined_stats = {
            **self.stats,
            **self.graph_manager.get_stats(),
        }
        
        # Calculate token-level hit rate
        if self.stats["total_tokens"] > 0:
            combined_stats["token_hit_rate"] = (
                self.stats["graph_tokens"] / self.stats["total_tokens"]
            )
        
        return combined_stats
    
    def clear_graphs(self):
        """Clear all captured graphs."""
        self.graph_manager.clear_graphs()
        logger.info("All model graphs cleared")


def wrap_model_with_cuda_graphs(
    model: nn.Module,
    capture_seq_lens: List[int] = None
) -> LlamaModelGraphWrapper:
    """
    Convenience function to wrap a model with CUDA graph support.
    
    Args:
        model: The model to wrap
        capture_seq_lens: Sequence lengths to capture graphs for
        
    Returns:
        Wrapped model with CUDA graph support
    """
    return LlamaModelGraphWrapper(model, capture_seq_lens=capture_seq_lens)