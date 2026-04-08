"""
Enhanced Context Length Bucket Management for CUDA Graph Compilation.

Phase 1, Task 1.4: Handle multiple context length buckets (512,1024,2048,4096)
in production with dynamic graph selection, memory-aware caching, and LRU eviction.
"""

import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import time
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import OrderedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Types of CUDA graphs for different context lengths."""
    SHORT = 512
    STANDARD = 1024
    MEDIUM = 2048
    LONG = 4096
    
    @classmethod
    def from_seq_len(cls, seq_len: int) -> Optional['GraphType']:
        """Get the appropriate graph type for a sequence length."""
        for graph_type in cls:
            if seq_len <= graph_type.value:
                return graph_type
        return None
    
    @classmethod
    def get_all_types(cls) -> List['GraphType']:
        """Get all graph types in order of increasing size."""
        return sorted(cls, key=lambda gt: gt.value)


@dataclass
class GraphInstance:
    """Enhanced graph instance with usage tracking."""
    graph_type: GraphType
    graph: torch.cuda.CUDAGraph
    input_nodes: Dict[str, torch.Tensor]
    output_nodes: Dict[str, torch.Tensor]
    capture_time: float
    memory_usage: int
    last_used: float = 0.0
    use_count: int = 0
    total_execution_time: float = 0.0


class GraphBucketManager:
    """
    Manages multiple context length buckets with LRU eviction.
    
    Features:
    1. Dynamic graph selection based on input size
    2. Memory-aware graph caching with LRU eviction
    3. Statistics for optimization decisions
    4. Automatic graph capture on demand
    5. Memory usage tracking and limits
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_total_memory_mb: int = 4096,  # 4GB default limit
        enable_lru: bool = True,
        capture_on_demand: bool = True
    ):
        """
        Initialize the graph bucket manager.
        
        Args:
            device: CUDA device to use
            max_total_memory_mb: Maximum memory for graphs (MB)
            enable_lru: Enable LRU eviction when memory limit reached
            capture_on_demand: Automatically capture missing graphs
        """
        self.device = torch.device(device)
        self.max_total_memory_bytes = max_total_memory_mb * 1024 * 1024
        self.enable_lru = enable_lru
        self.capture_on_demand = capture_on_demand
        
        # Graph storage with LRU tracking
        self.graphs: Dict[GraphType, GraphInstance] = OrderedDict()
        
        # Memory tracking
        self.total_memory_used = 0
        self.memory_by_type: Dict[GraphType, int] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "graph_hits": 0,
            "graph_misses": 0,
            "eager_fallbacks": 0,
            "graph_captures": 0,
            "graph_evictions": 0,
            "total_memory_allocated": 0,
            "peak_memory_used": 0,
            "average_graph_hit_time_ms": 0.0,
            "average_eager_time_ms": 0.0,
        }
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized GraphBucketManager on {self.device}")
        logger.info(f"Memory limit: {max_total_memory_mb} MB")
        logger.info(f"LRU eviction: {'enabled' if enable_lru else 'disabled'}")
        logger.info(f"Capture on demand: {'enabled' if capture_on_demand else 'disabled'}")
    
    def get_graph_for_seq_len(
        self,
        seq_len: int,
        capture_func=None,
        capture_args=None,
        capture_kwargs=None
    ) -> Optional[GraphInstance]:
        """
        Get the best fitting graph for a sequence length.
        
        Args:
            seq_len: Sequence length to match
            capture_func: Function to capture graph if not available
            capture_args: Arguments for capture function
            capture_kwargs: Keyword arguments for capture function
            
        Returns:
            GraphInstance or None if no suitable graph
        """
        self.stats["total_requests"] += 1
        
        # Find the best fitting graph type
        graph_type = self._find_best_fit_graph_type(seq_len)
        
        if graph_type is None:
            self.stats["graph_misses"] += 1
            logger.debug(f"No suitable graph for seq_len={seq_len}")
            return None
        
        # Check if we have this graph
        if graph_type in self.graphs:
            graph_instance = self.graphs[graph_type]
            
            # Update LRU order (move to end)
            self.graphs.move_to_end(graph_type)
            
            # Update usage statistics
            graph_instance.last_used = time.time()
            graph_instance.use_count += 1
            
            self.stats["graph_hits"] += 1
            logger.debug(f"Graph hit: {graph_type.name} for seq_len={seq_len}")
            
            return graph_instance
        
        # Graph not available
        self.stats["graph_misses"] += 1
        
        # Capture on demand if enabled
        if self.capture_on_demand and capture_func is not None:
            logger.info(f"Capturing graph on demand for {graph_type.name}")
            return self.capture_graph_on_demand(
                graph_type=graph_type,
                capture_func=capture_func,
                capture_args=capture_args or [],
                capture_kwargs=capture_kwargs or {}
            )
        
        logger.debug(f"Graph miss: {graph_type.name} not available for seq_len={seq_len}")
        return None
    
    def capture_graph_on_demand(
        self,
        graph_type: GraphType,
        capture_func,
        capture_args: List[Any],
        capture_kwargs: Dict[str, Any]
    ) -> Optional[GraphInstance]:
        """
        Capture a graph on demand, handling memory limits.
        
        Args:
            graph_type: Type of graph to capture
            capture_func: Function to capture
            capture_args: Arguments for capture function
            capture_kwargs: Keyword arguments for capture function
            
        Returns:
            GraphInstance or None if capture failed
        """
        # Check memory limits
        estimated_memory = self._estimate_graph_memory(graph_type)
        
        if self.total_memory_used + estimated_memory > self.max_total_memory_bytes:
            if self.enable_lru:
                logger.info(f"Memory limit reached, evicting LRU graphs...")
                if not self._evict_lru_graphs(estimated_memory):
                    logger.warning(f"Cannot free enough memory for {graph_type.name}")
                    return None
            else:
                logger.warning(f"Memory limit reached, cannot capture {graph_type.name}")
                return None
        
        try:
            # Capture the graph
            start_time = time.time()
            
            # Create sample inputs
            batch_size = 1
            seq_len = graph_type.value
            inputs = self._create_sample_inputs(batch_size, seq_len)
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = capture_func(*inputs, *capture_args, **capture_kwargs)
            
            torch.cuda.synchronize()
            
            # Capture graph
            graph = torch.cuda.CUDAGraph()
            static_inputs = self._make_static_tensors(inputs)
            
            with torch.cuda.graph(graph):
                static_outputs = capture_func(*static_inputs, *capture_args, **capture_kwargs)
            
            capture_time = time.time() - start_time
            
            # Calculate actual memory usage
            memory_usage = self._calculate_actual_memory(static_inputs, static_outputs)
            
            # Create graph instance
            graph_instance = GraphInstance(
                graph_type=graph_type,
                graph=graph,
                input_nodes={f"input_{i}": t for i, t in enumerate(static_inputs)},
                output_nodes={f"output_{i}": t for i, t in enumerate(static_outputs)},
                capture_time=capture_time,
                memory_usage=memory_usage,
                last_used=time.time(),
                use_count=1
            )
            
            # Store the graph
            self.graphs[graph_type] = graph_instance
            self.total_memory_used += memory_usage
            self.memory_by_type[graph_type] = memory_usage
            
            # Update statistics
            self.stats["graph_captures"] += 1
            self.stats["total_memory_allocated"] += memory_usage
            self.stats["peak_memory_used"] = max(
                self.stats["peak_memory_used"],
                self.total_memory_used
            )
            
            logger.info(f"Captured {graph_type.name} in {capture_time:.3f}s, "
                       f"memory: {memory_usage / 1024**2:.2f} MB")
            
            return graph_instance
            
        except Exception as e:
            logger.error(f"Failed to capture graph {graph_type.name}: {e}")
            return None
    
    def execute_with_bucket_selection(
        self,
        seq_len: int,
        eager_func,
        inputs: List[torch.Tensor],
        capture_func=None,
        capture_args=None,
        capture_kwargs=None
    ) -> Tuple[List[torch.Tensor], Optional[GraphType]]:
        """
        Execute with automatic bucket selection and fallback.
        
        Args:
            seq_len: Sequence length
            eager_func: Function for eager execution
            inputs: Input tensors
            capture_func: Optional function for graph capture
            capture_args: Arguments for capture function
            capture_kwargs: Keyword arguments for capture function
            
        Returns:
            Tuple of (outputs, graph_type_used)
        """
        start_time = time.time()
        
        # Try to get a graph
        graph_instance = self.get_graph_for_seq_len(
            seq_len=seq_len,
            capture_func=capture_func,
            capture_args=capture_args,
            capture_kwargs=capture_kwargs
        )
        
        if graph_instance is not None:
            # Execute with graph
            try:
                # Copy inputs to graph input nodes
                for i, (name, static_input) in enumerate(graph_instance.input_nodes.items()):
                    if i < len(inputs):
                        static_input.copy_(inputs[i])
                
                # Execute graph
                graph_instance.graph.replay()
                
                # Get outputs
                outputs = list(graph_instance.output_nodes.values())
                
                # Update execution time
                execution_time = time.time() - start_time
                graph_instance.total_execution_time += execution_time
                
                # Update performance stats
                self._update_performance_stats(
                    graph_type=graph_instance.graph_type,
                    execution_time=execution_time,
                    used_graph=True
                )
                
                return outputs, graph_instance.graph_type
                
            except Exception as e:
                logger.warning(f"Graph execution failed: {e}, falling back to eager")
                self.stats["eager_fallbacks"] += 1
        
        # Fall back to eager execution
        self.stats["eager_fallbacks"] += 1
        
        with torch.no_grad():
            outputs = eager_func(*inputs)
        
        execution_time = time.time() - start_time
        
        # Update performance stats
        self._update_performance_stats(
            graph_type=None,
            execution_time=execution_time,
            used_graph=False
        )
        
        return (outputs if isinstance(outputs, list) else [outputs]), None
    
    def pre_capture_graphs(
        self,
        capture_func,
        graph_types: Optional[List[GraphType]] = None,
        max_concurrent: int = 2
    ) -> Dict[GraphType, bool]:
        """
        Pre-capture graphs for specified types.
        
        Args:
            capture_func: Function to capture
            graph_types: List of graph types to capture (default: all)
            max_concurrent: Maximum concurrent captures
            
        Returns:
            Dictionary of capture results
        """
        if graph_types is None:
            graph_types = GraphType.get_all_types()
        
        results = {}
        
        for graph_type in graph_types:
            if graph_type in self.graphs:
                logger.info(f"Graph {graph_type.name} already captured")
                results[graph_type] = True
                continue
            
            # Check memory before capture
            estimated_memory = self._estimate_graph_memory(graph_type)
            
            if self.total_memory_used + estimated_memory > self.max_total_memory_bytes:
                if self.enable_lru:
                    if not self._evict_lru_graphs(estimated_memory):
                        logger.warning(f"Cannot free memory for {graph_type.name}")
                        results[graph_type] = False
                        continue
                else:
                    logger.warning(f"Memory limit reached, skipping {graph_type.name}")
                    results[graph_type] = False
                    continue
            
            # Capture the graph
            success = self._capture_single_graph(graph_type, capture_func)
            results[graph_type] = success
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        return {
            "total_memory_used_bytes": self.total_memory_used,
            "total_memory_used_mb": self.total_memory_used / (1024 * 1024),
            "memory_limit_bytes": self.max_total_memory_bytes,
            "memory_limit_mb": self.max_total_memory_bytes / (1024 * 1024),
            "memory_available_bytes": self.max_total_memory_bytes - self.total_memory_used,
            "memory_available_mb": (self.max_total_memory_bytes - self.total_memory_used) / (1024 * 1024),
            "memory_by_graph_type": {
                gt.name: mem / (1024 * 1024)
                for gt, mem in self.memory_by_type.items()
            },
            "graph_count": len(self.graphs),
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        total_executions = self.stats["graph_hits"] + self.stats["eager_fallbacks"]
        
        if total_executions > 0:
            hit_rate = self.stats["graph_hits"] / total_executions * 100
        else:
            hit_rate = 0.0
        
        # Calculate average times
        avg_graph_time = 0.0
        avg_eager_time = 0.0
        
        if len(self.performance_history) > 0:
            graph_times = [h["execution_time"] for h in self.performance_history if h["used_graph"]]
            eager_times = [h["execution_time"] for h in self.performance_history if not h["used_graph"]]
            
            if graph_times:
                avg_graph_time = sum(graph_times) / len(graph_times) * 1000  # ms
            if eager_times:
                avg_eager_time = sum(eager_times) / len(eager_times) * 1000  # ms
        
        return {
            "total_requests": self.stats["total_requests"],
            "graph_hit_rate_percent": hit_rate,
            "graph_hits": self.stats["graph_hits"],
            "graph_misses": self.stats["graph_misses"],
            "eager_fallbacks": self.stats["eager_fallbacks"],
            "graph_captures": self.stats["graph_captures"],
            "graph_evictions": self.stats["graph_evictions"],
            "average_graph_time_ms": avg_graph_time,
            "average_eager_time_ms": avg_eager_time,
            "speedup_factor": avg_eager_time / avg_graph_time if avg_graph_time > 0 else 0,
            "memory_efficiency_mb_per_graph": self.total_memory_used / max(len(self.graphs), 1) / (1024 * 1024),
        }
    
    def clear_all_graphs(self):
        """Clear all graphs and reset memory tracking."""
        self.graphs.clear()
        self.total_memory_used = 0
        self.memory_by_type.clear()
        logger.info("All graphs cleared")
    
    def _find_best_fit_graph_type(self, seq_len: int) -> Optional[GraphType]:
        """Find the best fitting graph type for a sequence length."""
        # First, check if we have an exact match
        for graph_type in self.graphs.keys():
            if seq_len <= graph_type.value:
                # Among suitable graphs, pick the one with smallest capacity
                suitable = [gt for gt in self.graphs.keys() if seq_len <= gt.value]
                if suitable:
                    return min(suitable, key=lambda gt: gt.value)
        
        # No suitable graph captured
        return GraphType.from_seq_len(seq_len)
    
    def _evict_lru_graphs(self, required_memory: int) -> bool:
        """
        Evict least recently used graphs to free memory.
        
        Args:
            required_memory: Memory needed (bytes)
            
        Returns:
            True if enough memory was freed, False otherwise
        """
        memory_freed = 0
        graphs_to_evict = []
        
        # Collect LRU graphs until we have enough memory
        for graph_type, graph_instance in self.graphs.items():
            if memory_freed >= required_memory:
                break
            
            graphs_to_evict.append(graph_type)
            memory_freed += graph_instance.memory_usage
        
        # Check if we can free enough memory
        if memory_freed < required_memory:
            logger.warning(f"Cannot free enough memory: need {required_memory}, can free {memory_freed}")
            return False
        
        # Evict the graphs
        for graph_type in graphs_to_evict:
            graph_instance = self.graphs.pop(graph_type)
            self.total_memory_used -= graph_instance.memory_usage
            self.memory_by_type.pop(graph_type, None)
            self.stats["graph_evictions"] += 1
            
            logger.info(f"Evicted {graph_type.name} (LRU), "
                       f"freed {graph_instance.memory_usage / 1024**2:.2f} MB")
        
        return True
    
    def _estimate_graph_memory(self, graph_type: GraphType) -> int:
        """Estimate memory needed for a graph type."""
        # Simple estimation based on context length
        base_memory = 50 * 1024 * 1024  # 50 MB base
        memory_per_token = 1024  # 1 KB per token (simplified)
        
        return base_memory + (graph_type.value * memory_per_token)
    
    def _calculate_actual_memory(
        self,
        inputs: List[Any],
        outputs: List[Any]
    ) -> int:
        """Calculate actual memory usage of tensors."""
        total_memory = 0
        
        def add_tensor_memory(tensor):
            nonlocal total_memory
            if isinstance(tensor, torch.Tensor):
                total_memory += tensor.numel() * tensor.element_size()
        
        # Input memory
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                add_tensor_memory(tensor)
            elif isinstance(tensor, list):
                for item in tensor:
                    if isinstance(item, tuple):
                        for t in item:
                            add_tensor_memory(t)
        
        # Output memory
        for tensor in outputs:
            if isinstance(tensor, torch.Tensor):
                add_tensor_memory(tensor)
            elif isinstance(tensor, list):
                for item in tensor:
                    if isinstance(item, tuple):
                        for t in item:
                            add_tensor_memory(t)
        
        # Add graph structure overhead
        total_memory += 10 * 1024 * 1024  # 10 MB overhead
        
        return total_memory
    
    def _create_sample_inputs(self, batch_size: int, seq_len: int) -> List[torch.Tensor]:
        """Create sample inputs for graph capture."""
        # Input IDs
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
        
        # Attention mask
        attention_mask = torch.ones((batch_size, seq_len), device=self.device, dtype=torch.bool)
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Past key values (simplified for capture)
        hidden_size = 4096
        num_heads = 32
        head_dim = 128
        
        past_key_values = []
        for _ in range(32):  # 32 layers
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
                static_tensor = torch.empty_like(tensor, device=self.device)
                static_tensors.append(static_tensor)
            elif isinstance(tensor, list):
                static_list = []
                for item in tensor:
                    if isinstance(item, tuple):
                        static_key = torch.empty_like(item[0], device=self.device)
                        static_value = torch.empty_like(item[1], device=self.device)
                        static_list.append((static_key, static_value))
                static_tensors.append(static_list)
            else:
                raise TypeError(f"Unsupported type: {type(tensor)}")
        
        return static_tensors
    
    def _capture_single_graph(
        self,
        graph_type: GraphType,
        capture_func
    ) -> bool:
        """Capture a single graph."""
        try:
            # This is a simplified version - in practice would use capture_graph_on_demand
            batch_size = 1
            seq_len = graph_type.value
            inputs = self._create_sample_inputs(batch_size, seq_len)
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = capture_func(*inputs)
            
            torch.cuda.synchronize()
            
            # Capture
            graph = torch.cuda.CUDAGraph()
            static_inputs = self._make_static_tensors(inputs)
            
            with torch.cuda.graph(graph):
                static_outputs = capture_func(*static_inputs)
            
            # Create instance
            memory_usage = self._calculate_actual_memory(static_inputs, static_outputs)
            
            graph_instance = GraphInstance(
                graph_type=graph_type,
                graph=graph,
                input_nodes={f"input_{i}": t for i, t in enumerate(static_inputs)},
                output_nodes={f"output_{i}": t for i, t in enumerate(static_outputs)},
                capture_time=0.0,  # Would measure in real implementation
                memory_usage=memory_usage,
                last_used=time.time(),
                use_count=0
            )
            
            # Store
            self.graphs[graph_type] = graph_instance
            self.total_memory_used += memory_usage
            self.memory_by_type[graph_type] = memory_usage
            
            logger.info(f"Pre-captured {graph_type.name}, "
                       f"memory: {memory_usage / 1024**2:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pre-capture {graph_type.name}: {e}")
            return False
    
    def _update_performance_stats(
        self,
        graph_type: Optional[GraphType],
        execution_time: float,
        used_graph: bool
    ):
        """Update performance statistics."""
        self.performance_history.append({
            "timestamp": time.time(),
            "graph_type": graph_type.name if graph_type else None,
            "execution_time": execution_time,
            "used_graph": used_graph,
        })
        
        # Keep only recent history (last 1000 executions)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]


class AdaptiveBucketManager(GraphBucketManager):
    """
    Enhanced bucket manager with adaptive learning.
    
    Features:
    1. Learns optimal graph selection based on usage patterns
    2. Adaptive memory allocation based on hit rates
    3. Predictive graph pre-capture based on patterns
    4. Dynamic bucket size adjustment
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Learning data
        self.usage_patterns: Dict[int, int] = {}  # seq_len -> count
        self.hit_patterns: Dict[GraphType, Dict[str, Any]] = {}
        
        # Adaptive settings
        self.adaptive_memory_allocation = kwargs.get('adaptive_memory_allocation', True)
        self.learning_window = kwargs.get('learning_window', 1000)
        
        # Initialize hit patterns
        for graph_type in GraphType.get_all_types():
            self.hit_patterns[graph_type] = {
                "hits": 0,
                "misses": 0,
                "average_seq_len": 0,
                "last_used": 0.0,
            }
    
    def get_graph_for_seq_len(self, *args, **kwargs):
        """Override with learning."""
        seq_len = args[0] if args else kwargs.get('seq_len')
        
        # Record usage pattern
        self.usage_patterns[seq_len] = self.usage_patterns.get(seq_len, 0) + 1
        
        # Call parent method
        result = super().get_graph_for_seq_len(*args, **kwargs)
        
        # Update hit patterns
        if result is not None:
            graph_type = result.graph_type
            patterns = self.hit_patterns[graph_type]
            patterns["hits"] += 1
            patterns["last_used"] = time.time()
            
            # Update average sequence length (exponential moving average)
            old_avg = patterns["average_seq_len"]
            if old_avg == 0:
                patterns["average_seq_len"] = seq_len
            else:
                patterns["average_seq_len"] = 0.9 * old_avg + 0.1 * seq_len
        else:
            # Record miss for the graph type that would have been used
            graph_type = GraphType.from_seq_len(seq_len)
            if graph_type:
                self.hit_patterns[graph_type]["misses"] += 1
        
        return result
    
    def get_optimal_graphs_to_keep(self) -> List[GraphType]:
        """Determine which graphs to keep based on usage patterns."""
        if not self.adaptive_memory_allocation:
            return list(self.graphs.keys())
        
        # Calculate utility score for each graph type
        utility_scores = {}
        
        for graph_type in GraphType.get_all_types():
            patterns = self.hit_patterns[graph_type]
            total_uses = patterns["hits"] + patterns["misses"]
            
            if total_uses == 0:
                utility_scores[graph_type] = 0
                continue
            
            # Calculate utility based on hit rate and recency
            hit_rate = patterns["hits"] / total_uses
            recency = 1.0 / (time.time() - patterns["last_used"] + 1)  # Avoid division by zero
            
            # Memory efficiency (inverse of memory usage)
            memory_mb = self.memory_by_type.get(graph_type, 0) / (1024 * 1024)
            memory_efficiency = 1.0 / (memory_mb + 1)  # Avoid division by zero
            
            # Combined utility score
            utility = hit_rate * 0.5 + recency * 0.3 + memory_efficiency * 0.2
            utility_scores[graph_type] = utility
        
        # Sort by utility (descending)
        sorted_graphs = sorted(
            utility_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Determine which graphs to keep based on memory limits
        kept_graphs = []
        current_memory = 0
        
        for graph_type, utility in sorted_graphs:
            if graph_type not in self.graphs:
                continue
            
            graph_memory = self.memory_by_type[graph_type]
            
            if current_memory + graph_memory <= self.max_total_memory_bytes:
                kept_graphs.append(graph_type)
                current_memory += graph_memory
            else:
                # Would need to evict this graph
                pass
        
        return kept_graphs
    
    def optimize_memory_allocation(self):
        """Reallocate memory based on usage patterns."""
        if not self.adaptive_memory_allocation:
            return
        
        logger.info("Optimizing memory allocation based on usage patterns...")
        
        # Get optimal graphs to keep
        optimal_graphs = self.get_optimal_graphs_to_keep()
        
        # Evict non-optimal graphs
        graphs_to_evict = [
            gt for gt in self.graphs.keys()
            if gt not in optimal_graphs
        ]
        
        for graph_type in graphs_to_evict:
            graph_instance = self.graphs.pop(graph_type)
            self.total_memory_used -= graph_instance.memory_usage
            self.memory_by_type.pop(graph_type, None)
            self.stats["graph_evictions"] += 1
            
            logger.info(f"Evicted {graph_type.name} (optimization), "
                       f"freed {graph_instance.memory_usage / 1024**2:.2f} MB")
        
        # Report optimization results
        logger.info(f"Memory optimization complete. Keeping {len(optimal_graphs)} graphs.")
        logger.info(f"Total memory used: {self.total_memory_used / 1024**2:.2f} MB")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate a learning report."""
        total_uses = sum(self.usage_patterns.values())
        
        # Most common sequence lengths
        common_seq_lens = sorted(
            self.usage_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Graph type effectiveness
        graph_effectiveness = {}
        for graph_type, patterns in self.hit_patterns.items():
            total = patterns["hits"] + patterns["misses"]
            if total > 0:
                effectiveness = patterns["hits"] / total * 100
                graph_effectiveness[graph_type.name] = {
                    "effectiveness_percent": effectiveness,
                    "hits": patterns["hits"],
                    "misses": patterns["misses"],
                    "average_seq_len": patterns["average_seq_len"],
                }
        
        return {
            "total_requests_analyzed": total_uses,
            "most_common_sequence_lengths": common_seq_lens,
            "graph_effectiveness": graph_effectiveness,
            "memory_allocation_optimization": "enabled" if self.adaptive_memory_allocation else "disabled",
            "recommended_graphs_to_keep": [gt.name for gt in self.get_optimal_graphs_to_keep()],
        }


def create_production_bucket_manager(
    device: str = "cuda",
    memory_limit_mb: int = 4096,
    adaptive: bool = True
) -> GraphBucketManager:
    """
    Factory function to create a production-ready bucket manager.
    
    Args:
        device: CUDA device
        memory_limit_mb: Memory limit in MB
        adaptive: Use adaptive learning manager
        
    Returns:
        Configured bucket manager
    """
    if adaptive:
        return AdaptiveBucketManager(
            device=device,
            max_total_memory_mb=memory_limit_mb,
            enable_lru=True,
            capture_on_demand=True,
            adaptive_memory_allocation=True
        )
    else:
        return GraphBucketManager(
            device=device,
            max_total_memory_mb=memory_limit_mb,
            enable_lru=True,
            capture_on_demand=True
        )