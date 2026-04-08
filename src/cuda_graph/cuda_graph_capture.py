"""
CUDA Graph Capture Implementation for LLM Decode Loops.

This module implements static CUDA graph capture to eliminate per-token
dispatch overhead in LLM inference (Phase 1, Task 1.3).
"""

import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Types of CUDA graphs for different context lengths."""
    SHORT = 512
    STANDARD = 1024
    MEDIUM = 2048
    LONG = 4096


@dataclass
class GraphInstance:
    """Represents a captured CUDA graph instance."""
    graph_type: GraphType
    graph: torch.cuda.CUDAGraph
    input_nodes: Dict[str, torch.Tensor]
    output_nodes: Dict[str, torch.Tensor]
    capture_time: float
    memory_usage: int


class CUDAGraphManager:
    """
    Manages CUDA graph compilation and execution for LLM decode loops.
    
    This class handles:
    1. Graph capture for different context length buckets
    2. Graph instantiation and execution
    3. Memory management for graph inputs/outputs
    4. Fallback to eager execution when graphs don't match
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the CUDA graph manager.
        
        Args:
            device: CUDA device to use (default: "cuda")
        """
        self.device = torch.device(device)
        self.graphs: Dict[GraphType, GraphInstance] = {}
        self.stream = torch.cuda.Stream()
        self._is_capturing = False
        self._capture_context = None
        
        # Statistics
        self.stats = {
            "graph_executions": 0,
            "eager_executions": 0,
            "graph_hit_rate": 0.0,
            "total_time_saved": 0.0,
        }
        
        logger.info(f"Initialized CUDA Graph Manager on device: {self.device}")
    
    def capture_graph(
        self,
        graph_type: GraphType,
        capture_func,
        *args,
        **kwargs
    ) -> GraphInstance:
        """
        Capture a CUDA graph for a specific context length.
        
        Args:
            graph_type: Type of graph (context length bucket)
            capture_func: Function that performs the computation to capture
            *args, **kwargs: Arguments to pass to capture_func
            
        Returns:
            GraphInstance containing the captured graph
        """
        logger.info(f"Capturing CUDA graph for context length: {graph_type.value}")
        
        # Create input/output tensors with the right shapes
        # For LLM decode, we need:
        # - input_ids: [batch_size, seq_len]
        # - attention_mask: [batch_size, seq_len]
        # - position_ids: [batch_size, seq_len]
        # - past_key_values: list of tuples
        
        batch_size = 1  # Typical for inference
        seq_len = graph_type.value
        
        # Create sample inputs for capture
        inputs = self._create_sample_inputs(batch_size, seq_len)
        
        # Warm up before capture
        logger.info("Warming up before graph capture...")
        with torch.no_grad():
            for _ in range(3):
                _ = capture_func(*inputs)
        
        torch.cuda.synchronize()
        
        # Capture the graph
        logger.info("Starting CUDA graph capture...")
        start_time = time.time()
        
        # Create a new graph
        graph = torch.cuda.CUDAGraph()
        
        # Create static inputs/outputs
        static_inputs = self._make_static_tensors(inputs)
        static_outputs = None
        
        # Capture the graph
        with torch.cuda.graph(graph):
            # Execute the capture function with static inputs
            static_outputs = capture_func(*static_inputs)
        
        capture_time = time.time() - start_time
        
        # Estimate memory usage
        memory_usage = self._estimate_graph_memory(graph, static_inputs, static_outputs)
        
        # Create graph instance
        graph_instance = GraphInstance(
            graph_type=graph_type,
            graph=graph,
            input_nodes={f"input_{i}": t for i, t in enumerate(static_inputs)},
            output_nodes={f"output_{i}": t for i, t in enumerate(static_outputs)},
            capture_time=capture_time,
            memory_usage=memory_usage
        )
        
        # Store the graph
        self.graphs[graph_type] = graph_instance
        
        logger.info(f"Graph captured successfully in {capture_time:.3f}s")
        logger.info(f"Estimated memory usage: {memory_usage / 1024**2:.2f} MB")
        
        return graph_instance
    
    def execute_graph(
        self,
        graph_type: GraphType,
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Execute a captured CUDA graph.
        
        Args:
            graph_type: Type of graph to execute
            inputs: List of input tensors
            
        Returns:
            List of output tensors
        """
        if graph_type not in self.graphs:
            raise ValueError(f"No graph captured for type: {graph_type}")
        
        graph_instance = self.graphs[graph_type]
        
        # Copy inputs to graph input nodes
        for i, (name, static_input) in enumerate(graph_instance.input_nodes.items()):
            if i < len(inputs):
                static_input.copy_(inputs[i])
        
        # Execute the graph
        graph_instance.graph.replay()
        
        # Get outputs
        outputs = list(graph_instance.output_nodes.values())
        
        # Update statistics
        self.stats["graph_executions"] += 1
        self._update_stats()
        
        return outputs
    
    def get_best_fit_graph(self, seq_len: int) -> Optional[GraphType]:
        """
        Find the best fitting graph for a given sequence length.
        
        Args:
            seq_len: Sequence length to match
            
        Returns:
            Best fitting GraphType or None if no suitable graph
        """
        # Find the smallest graph that can accommodate the sequence length
        suitable_graphs = [
            gt for gt in self.graphs.keys()
            if gt.value >= seq_len
        ]
        
        if not suitable_graphs:
            return None
        
        # Return the smallest suitable graph
        return min(suitable_graphs, key=lambda gt: gt.value)
    
    def execute_with_graph_fallback(
        self,
        seq_len: int,
        eager_func,
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Execute with graph if available, otherwise fall back to eager execution.
        
        Args:
            seq_len: Sequence length of the current operation
            eager_func: Function to call for eager execution
            inputs: Input tensors
            
        Returns:
            Output tensors
        """
        # Try to find a suitable graph
        graph_type = self.get_best_fit_graph(seq_len)
        
        if graph_type is not None:
            try:
                return self.execute_graph(graph_type, inputs)
            except Exception as e:
                logger.warning(f"Graph execution failed, falling back to eager: {e}")
                self.stats["eager_executions"] += 1
        
        # Fall back to eager execution
        self.stats["eager_executions"] += 1
        with torch.no_grad():
            outputs = eager_func(*inputs)
        
        self._update_stats()
        return outputs if isinstance(outputs, list) else [outputs]
    
    def _create_sample_inputs(self, batch_size: int, seq_len: int) -> List[torch.Tensor]:
        """Create sample inputs for graph capture."""
        # Input IDs
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
        
        # Attention mask (all ones for decode)
        attention_mask = torch.ones((batch_size, seq_len), device=self.device, dtype=torch.bool)
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # For simplicity in capture, we'll use minimal past_key_values
        # In real implementation, this would match the model architecture
        hidden_size = 4096  # Typical for 32B model
        num_heads = 32
        head_dim = 128
        
        # Create dummy past key values
        past_key_values = []
        for _ in range(32):  # 32 layers typical for 32B model
            past_key = torch.randn(
                batch_size, num_heads, seq_len - 1, head_dim,
                device=self.device
            )
            past_value = torch.randn(
                batch_size, num_heads, seq_len - 1, head_dim,
                device=self.device
            )
            past_key_values.append((past_key, past_value))
        
        return [input_ids, attention_mask, position_ids, past_key_values]
    
    def _make_static_tensors(self, tensors: List[Any]) -> List[torch.Tensor]:
        """Convert tensors to static versions for graph capture."""
        static_tensors = []
        
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                # Create a static copy
                static_tensor = torch.empty_like(tensor, device=self.device)
                static_tensors.append(static_tensor)
            elif isinstance(tensor, list):
                # Handle lists (like past_key_values)
                static_list = []
                for item in tensor:
                    if isinstance(item, tuple):
                        # Handle (key, value) tuples
                        static_key = torch.empty_like(item[0], device=self.device)
                        static_value = torch.empty_like(item[1], device=self.device)
                        static_list.append((static_key, static_value))
                static_tensors.append(static_list)
            else:
                raise TypeError(f"Unsupported type for graph capture: {type(tensor)}")
        
        return static_tensors
    
    def _estimate_graph_memory(
        self,
        graph: torch.cuda.CUDAGraph,
        inputs: List[Any],
        outputs: List[Any]
    ) -> int:
        """Estimate memory usage of a captured graph."""
        # This is a simplified estimation
        # In practice, would need to query CUDA memory info
        
        total_memory = 0
        
        # Estimate input memory
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                total_memory += tensor.numel() * tensor.element_size()
            elif isinstance(tensor, list):
                for item in tensor:
                    if isinstance(item, tuple):
                        for t in item:
                            if isinstance(t, torch.Tensor):
                                total_memory += t.numel() * t.element_size()
        
        # Estimate output memory
        for tensor in outputs:
            if isinstance(tensor, torch.Tensor):
                total_memory += tensor.numel() * tensor.element_size()
            elif isinstance(tensor, list):
                for item in tensor:
                    if isinstance(item, tuple):
                        for t in item:
                            if isinstance(t, torch.Tensor):
                                total_memory += t.numel() * t.element_size()
        
        # Add overhead for graph structure (rough estimate)
        total_memory += 50 * 1024 * 1024  # 50 MB overhead
        
        return total_memory
    
    def _update_stats(self):
        """Update statistics."""
        total_executions = self.stats["graph_executions"] + self.stats["eager_executions"]
        if total_executions > 0:
            self.stats["graph_hit_rate"] = self.stats["graph_executions"] / total_executions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.copy()
    
    def clear_graphs(self):
        """Clear all captured graphs."""
        self.graphs.clear()
        logger.info("All CUDA graphs cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_graphs()


def create_context_length_buckets() -> Dict[GraphType, int]:
    """
    Create context length buckets for graph compilation.
    
    Returns:
        Dictionary mapping GraphType to maximum tokens
    """
    return {
        GraphType.SHORT: 512,
        GraphType.STANDARD: 1024,
        GraphType.MEDIUM: 2048,
        GraphType.LONG: 4096,
    }


def benchmark_graph_vs_eager(
    graph_manager: CUDAGraphManager,
    eager_func,
    seq_len: int,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark graph execution vs eager execution.
    
    Args:
        graph_manager: CUDA graph manager instance
        eager_func: Function for eager execution
        seq_len: Sequence length to test
        num_iterations: Number of iterations to run
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking graph vs eager for seq_len={seq_len}")
    
    # Create test inputs
    batch_size = 1
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=graph_manager.device)
    attention_mask = torch.ones((batch_size, seq_len), device=graph_manager.device, dtype=torch.bool)
    position_ids = torch.arange(seq_len, device=graph_manager.device).unsqueeze(0).expand(batch_size, -1)
    
    # Simple dummy past_key_values for benchmarking
    past_key_values = []
    
    inputs = [input_ids, attention_mask, position_ids, past_key_values]
    
    # Warm up
    logger.info("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = eager_func(*inputs)
    
    torch.cuda.synchronize()
    
    # Benchmark eager execution
    logger.info("Benchmarking eager execution...")
    eager_times = []
    
    for i in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = eager_func(*inputs)
        torch.cuda.synchronize()
        eager_times.append(time.time() - start)
    
    eager_avg = sum(eager_times) / len(eager_times)
    eager_std = (sum((t - eager_avg) ** 2 for t in eager_times) / len(eager_times)) ** 0.5
    
    # Benchmark graph execution (if graph available)
    graph_type = graph_manager.get_best_fit_graph(seq_len)
    graph_results = {}
    
    if graph_type is not None:
        logger.info(f"Benchmarking graph execution ({graph_type.name})...")
        graph_times = []
        
        for i in range(num_iterations):
            start = time.time()
            _ = graph_manager.execute_graph(graph_type, inputs)
            torch.cuda.synchronize()
            graph_times.append(time.time() - start)
        
        graph_avg = sum(graph_times) / len(graph_times)
        graph_std = (sum((t - graph_avg) ** 2 for t in graph_times) / len(graph_times)) ** 0.5
        
        # Calculate speedup
        speedup = eager_avg / graph_avg if graph_avg > 0 else 0
        
        graph_results = {
            "graph_avg_ms": graph_avg * 1000,
            "graph_std_ms": graph_std * 1000,
            "speedup": speedup,
            "graph_type": graph_type.name,
        }
    
    results = {
        "seq_len": seq_len,
        "eager_avg_ms": eager_avg * 1000,
        "eager_std_ms": eager_std * 1000,
        "num_iterations": num_iterations,
        **graph_results,
    }
    
    logger.info(f"Benchmark results: {results}")
    return results