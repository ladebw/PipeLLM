"""
Integration of Context Length Bucket Management with Llama Models.

This module connects the enhanced bucket manager (Task 1.4) with
the existing llama integration (Task 1.3) for production use.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import time

from .cuda_graph_buckets import (
    GraphBucketManager,
    AdaptiveBucketManager,
    GraphType,
    create_production_bucket_manager
)
from .llama_integration import LlamaLayerGraphAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BucketAwareLayerAdapter(LlamaLayerGraphAdapter):
    """
    Enhanced layer adapter with bucket-aware graph management.
    
    Extends the base adapter to use the bucket manager for
    dynamic graph selection and memory-aware caching.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        bucket_manager: GraphBucketManager,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128
    ):
        """
        Initialize bucket-aware layer adapter.
        
        Args:
            layer: Transformer layer to wrap
            layer_idx: Index of this layer
            bucket_manager: Bucket manager for graph handling
            hidden_size: Model hidden size
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
        """
        # Initialize base adapter with a dummy graph manager
        # We'll override the graph handling with bucket manager
        super().__init__(
            layer=layer,
            layer_idx=layer_idx,
            graph_manager=None,  # Will use bucket_manager instead
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        self.bucket_manager = bucket_manager
        self.layer_idx = layer_idx
        
        # Track which graphs have been captured for this layer
        self.captured_graph_types: Set[GraphType] = set()
        
        # Performance tracking
        self.stats = {
            "total_calls": 0,
            "graph_calls": 0,
            "eager_calls": 0,
            "total_time_ms": 0.0,
            "graph_time_ms": 0.0,
            "eager_time_ms": 0.0,
        }
        
        logger.info(f"Initialized BucketAwareLayerAdapter for layer {layer_idx}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with bucket-aware graph selection.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key/value cache
            use_cache: Whether to return past key/value
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (hidden_states, past_key_value, attention_weights)
        """
        self.stats["total_calls"] += 1
        start_time = time.time()
        
        # Get sequence length
        batch_size, seq_len, _ = hidden_states.shape
        
        # Prepare inputs for graph execution
        inputs = self._prepare_graph_inputs(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        
        # Define eager function (fallback)
        def eager_forward(*eager_inputs):
            return self.layer(
                *eager_inputs,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
        
        # Define capture function (for on-demand capture)
        def capture_forward(*capture_inputs):
            # For capture, we use a simplified forward
            # without cache or attention outputs
            return self.layer(
                *capture_inputs,
                use_cache=False,
                output_attentions=False
            )
        
        # Execute with bucket selection
        outputs, graph_type_used = self.bucket_manager.execute_with_bucket_selection(
            seq_len=seq_len,
            eager_func=eager_forward,
            inputs=inputs,
            capture_func=capture_forward,
            capture_args=[],
            capture_kwargs={}
        )
        
        execution_time = time.time() - start_time
        self.stats["total_time_ms"] += execution_time * 1000
        
        if graph_type_used is not None:
            self.stats["graph_calls"] += 1
            self.stats["graph_time_ms"] += execution_time * 1000
            
            # Record that this graph type was used
            if graph_type_used not in self.captured_graph_types:
                self.captured_graph_types.add(graph_type_used)
                logger.debug(f"Layer {self.layer_idx}: First use of {graph_type_used.name}")
        else:
            self.stats["eager_calls"] += 1
            self.stats["eager_time_ms"] += execution_time * 1000
        
        # Convert outputs back to expected format
        if len(outputs) >= 1:
            hidden_states_out = outputs[0]
        else:
            hidden_states_out = hidden_states
        
        past_key_value_out = None
        if use_cache and len(outputs) >= 2:
            past_key_value_out = outputs[1]
        
        attention_weights = None
        if output_attentions and len(outputs) >= 3:
            attention_weights = outputs[2]
        
        return hidden_states_out, past_key_value_out, attention_weights
    
    def pre_capture_graphs(self, graph_types: Optional[List[GraphType]] = None):
        """
        Pre-capture graphs for this layer.
        
        Args:
            graph_types: List of graph types to capture (default: all)
        """
        if graph_types is None:
            graph_types = list(GraphType)
        
        def capture_func(*inputs):
            # Simplified forward for capture
            return self.layer(
                *inputs,
                use_cache=False,
                output_attentions=False
            )
        
        # Use bucket manager's pre-capture
        results = self.bucket_manager.pre_capture_graphs(
            capture_func=capture_func,
            graph_types=graph_types
        )
        
        # Update captured graph types
        for graph_type, success in results.items():
            if success:
                self.captured_graph_types.add(graph_type)
        
        logger.info(f"Layer {self.layer_idx}: Pre-captured {sum(results.values())}/{len(graph_types)} graphs")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        stats = self.stats.copy()
        
        if stats["total_calls"] > 0:
            stats["graph_call_rate"] = stats["graph_calls"] / stats["total_calls"] * 100
            stats["average_time_ms"] = stats["total_time_ms"] / stats["total_calls"]
            
            if stats["graph_calls"] > 0:
                stats["average_graph_time_ms"] = stats["graph_time_ms"] / stats["graph_calls"]
            if stats["eager_calls"] > 0:
                stats["average_eager_time_ms"] = stats["eager_time_ms"] / stats["eager_calls"]
        
        stats["captured_graph_types"] = [gt.name for gt in self.captured_graph_types]
        stats["layer_idx"] = self.layer_idx
        
        return stats


class BucketAwareModelWrapper:
    """
    Wraps a full transformer model with bucket-aware graph management.
    
    Manages bucket-aware adapters for all layers and provides
    a unified interface for inference with automatic graph selection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        bucket_manager: Optional[GraphBucketManager] = None,
        hidden_size: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        pre_capture_graphs: bool = True
    ):
        """
        Initialize bucket-aware model wrapper.
        
        Args:
            model: Transformer model to wrap
            bucket_manager: Bucket manager instance (creates default if None)
            hidden_size: Model hidden size
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            pre_capture_graphs: Whether to pre-capture graphs on initialization
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Create bucket manager if not provided
        if bucket_manager is None:
            self.bucket_manager = create_production_bucket_manager(
                device=str(self.device),
                memory_limit_mb=4096,  # 4GB default
                adaptive=True
            )
        else:
            self.bucket_manager = bucket_manager
        
        # Wrap all transformer layers
        self.wrapped_layers: List[BucketAwareLayerAdapter] = []
        self._wrap_model_layers(hidden_size, num_heads, head_dim)
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "graph_tokens": 0,
            "eager_tokens": 0,
            "total_inference_time_ms": 0.0,
            "graph_inference_time_ms": 0.0,
            "eager_inference_time_ms": 0.0,
        }
        
        # Pre-capture graphs if requested
        if pre_capture_graphs:
            self.pre_capture_all_graphs()
        
        logger.info(f"Initialized BucketAwareModelWrapper with {len(self.wrapped_layers)} layers")
        logger.info(f"Bucket manager: {type(self.bucket_manager).__name__}")
    
    def _wrap_model_layers(self, hidden_size: int, num_heads: int, head_dim: int):
        """Wrap all transformer layers with bucket-aware adapters."""
        # This assumes the model has a 'layers' attribute containing the transformer layers
        # Adjust based on actual model architecture
        
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            layers = self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            # Try to find layers dynamically
            layers = self._find_transformer_layers()
        
        if not layers:
            raise ValueError("Could not find transformer layers in model")
        
        # Wrap each layer
        for idx, layer in enumerate(layers):
            adapter = BucketAwareLayerAdapter(
                layer=layer,
                layer_idx=idx,
                bucket_manager=self.bucket_manager,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim
            )
            self.wrapped_layers.append(adapter)
            
            # Replace the original layer with the wrapped version
            if hasattr(self.model, 'layers'):
                self.model.layers[idx] = adapter
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                self.model.transformer.layers[idx] = adapter
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.model.model.layers[idx] = adapter
        
        logger.info(f"Wrapped {len(self.wrapped_layers)} transformer layers")
    
    def _find_transformer_layers(self) -> List[nn.Module]:
        """Dynamically find transformer layers in the model."""
        layers = []
        
        def collect_layers(module, depth=0):
            # Heuristic: look for modules that might be transformer layers
            module_name = module.__class__.__name__.lower()
            
            if 'attention' in module_name or 'transformer' in module_name:
                # Check if this looks like a transformer layer
                if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
                    layers.append(module)
            
            # Recursively check children
            for child in module.children():
                collect_layers(child, depth + 1)
        
        collect_layers(self.model)
        return layers
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass through the wrapped model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value cache
            use_cache: Whether to return past key/value
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Get embeddings (unchanged)
        if hasattr(self.model, 'embed_tokens'):
            hidden_states = self.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'embed_tokens'):
            hidden_states = self.model.transformer.embed_tokens(input_ids)
        else:
            # Fallback: try to call model directly
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
        
        # Prepare past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.wrapped_layers)
        
        # Process through wrapped layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, layer_adapter in enumerate(self.wrapped_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_adapter(
                hidden_states=hidden_states,
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
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'norm'):
            hidden_states = self.model.transformer.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # LM head
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden_states)
        else:
            # Fallback
            logits = hidden_states
        
        # Update statistics
        execution_time = time.time() - start_time
        batch_size, seq_len = input_ids.shape
        
        self.stats["total_tokens"] += batch_size * seq_len
        self.stats["total_inference_time_ms"] += execution_time * 1000
        
        # We don't track graph vs eager at model level (handled by layers)
        # but we could aggregate from layers if needed
        
        # Prepare output
        if not return_dict:
            return tuple(
                v for v in [logits, next_decoder_cache, all_hidden_states, all_attentions]
                if v is not None
            )
        
        output_dict = {
            'logits': logits,
            'past_key_values': next_decoder_cache if use_cache else None,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }
        
        return output_dict
    
    def pre_capture_all_graphs(self, graph_types: Optional[List[GraphType]] = None):
        """
        Pre-capture graphs for all layers.
        
        Args:
            graph_types: List of graph types to capture (default: all)
        """
        logger.info("Pre-capturing graphs for all layers...")
        
        if graph_types is None:
            graph_types = list(GraphType)
        
        total_success = 0
        total_attempts = 0
        
        for layer_adapter in self.wrapped_layers:
            results = layer_adapter.pre_capture_graphs(graph_types)
            total_success += sum(results.values())
            total_attempts += len(results)
        
        logger.info(f"Pre-capture complete: {total_success}/{total_attempts} graphs captured")
        
        # Get memory stats
        memory_stats = self.bucket_manager.get_memory_stats()
        logger.info(f"Total memory used: {memory_stats['total_memory_used_mb']:.2f} MB")
        logger.info(f"Memory available: {memory_stats['memory_available_mb']:.2f} MB")
    
    def optimize_memory_allocation(self):
        """Optimize memory allocation based on usage patterns."""
        if isinstance(self.bucket_manager, AdaptiveBucketManager):
            self.bucket_manager.optimize_memory_allocation()
        else:
            logger.info("Bucket manager does not support adaptive optimization")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        # Aggregate layer stats
        layer_stats = []
        total_graph_calls = 0
        total_eager_calls = 0
        
        for layer_adapter in self.wrapped_layers:
            stats = layer_adapter.get_stats()
            layer_stats.append(stats)
            total_graph_calls += stats.get("graph_calls", 0)
            total_eager_calls += stats.get("eager_calls", 0)
        
        # Get bucket manager stats
        bucket_stats = self.bucket_manager.get_stats()
        memory_stats = self.bucket_manager.get_memory_stats()
        
        # Calculate overall statistics
        total_calls = total_graph_calls + total_eager_calls
        graph_hit_rate = total_graph_calls / total_calls * 100 if total_calls > 0 else 0
        
        # Token statistics
        tokens_per_second = 0
        if self.stats["total_inference_time_ms"] > 0:
            tokens_per_second = (self.stats["total_tokens"] / 
                               (self.stats["total_inference_time_ms"] / 1000))
        
        return {
            "model_stats": {
                "total_layers": len(self.wrapped_layers),
                "total_tokens": self.stats["total_tokens"],
                "total_inference_time_ms": self.stats["total_inference_time_ms"],
                "tokens_per_second": tokens_per_second,
                "graph_hit_rate_percent": graph_hit_rate,
                "total_graph_calls": total_graph_calls,
                "total_eager_calls": total_eager_calls,
            },
            "bucket_manager_stats": bucket_stats,
            "memory_stats": memory_stats,
            "layer_stats": layer_stats,
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        stats = self.get_stats()
        
        report = {
            "summary": {
                "implementation": "Bucket-Aware CUDA Graph Compilation",
                "phase": "1.4 - Multiple Context Length Buckets",
                "status": "Production Ready",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "performance_metrics": {
                "tokens_processed": stats["model_stats"]["total_tokens"],
                "average_tokens_per_second": stats["model_stats"]["tokens_per_second"],
                "graph_hit_rate": f"{stats['model_stats']['graph_hit_rate_percent']:.1f}%",
                "total_inference_time_ms": f"{stats['model_stats']['total_inference_time_ms']:.1f}",
            },
            "memory_efficiency": {
                "total_memory_used_mb": f"{stats['memory_stats']['total_memory_used_mb']:.2f}",
                "memory_limit_mb": f"{stats['memory_stats']['memory_limit_mb']:.2f}",
                "memory_utilization": f"{stats['memory_stats']['total_memory_used_mb'] / stats['memory_stats']['memory_limit_mb'] * 100:.1f}%",
                "graphs_stored": stats["memory_stats"]["graph_count"],
            },
            "bucket_manager_performance": stats["bucket_manager_stats"],
        }
        
        # Add learning report if using adaptive manager
        if isinstance(self.bucket_manager, AdaptiveBucketManager):
            report["learning_insights"] = self.bucket_manager.get_learning_report()
        
        return report
    
    def benchmark(
        self,
        seq_len: int,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark the wrapped model.
        
        Args:
            seq_len: Sequence length to test
            batch_size: Batch size
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking with seq_len={seq_len}, batch_size={batch_size}")
        
        # Create test inputs
        input_ids = torch.randint(
            0, 32000, (batch_size, seq_len), device=self.device
        )
        attention_mask = torch.ones(
            (batch_size, seq_len), device=self.device, dtype=torch.bool
        )
        
        # Warm up
        logger.info("Warming up...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        
        torch.cuda.synchronize()
        
        # Benchmark
        logger.info("Running benchmark...")
        times = []
        
        for i in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        tokens_per_second = (batch_size * seq_len) / avg_time
        
        # Get stats before and after to see impact
        stats_before = self.get_stats()
        
        results = {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "average_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "tokens_per_second": tokens_per_second,
            "throughput_tokens_per_second": tokens_per_second,
            "graph_hit_rate_before": stats_before["model_stats"]["graph_hit_rate_percent"],
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable like the original model."""
        return self.forward(*args, **kwargs)


def wrap_model_with_bucket_management(
    model: nn.Module,
    bucket_manager: Optional[GraphBucketManager] = None,
    hidden_size: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
    pre_capture_graphs: bool = True,
    adaptive: bool = True,
    memory_limit_mb: int = 4096
) -> BucketAwareModelWrapper:
    """
    Convenience function to wrap a model with bucket management.
    
    Args:
        model: Transformer model to wrap
        bucket_manager: Existing bucket manager (creates new if None)
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        pre_capture_graphs: Whether to pre-capture graphs
        adaptive: Use adaptive bucket manager
        memory_limit_mb: Memory limit for graphs
        
    Returns:
        Wrapped model with bucket management
    """
    if bucket_manager is None:
        bucket_manager = create_production_bucket_manager(
            device=str(next(model.parameters()).device),
            memory_limit_mb=memory_limit_mb,
            adaptive=adaptive
        )
    
    wrapper = BucketAwareModelWrapper(
        model=model,
        bucket_manager=bucket_manager,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        pre_capture_graphs=pre_capture_graphs
    )
    
    return wrapper