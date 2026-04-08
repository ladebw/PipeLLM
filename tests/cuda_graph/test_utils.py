#!/usr/bin/env python3
"""
Test utilities for CUDA graph tests.

This module provides common utilities, fixtures, and helpers
for testing the CUDA graph implementation.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cuda_graph.cuda_graph_capture import CUDAGraphManager, GraphType
from cuda_graph.cuda_graph_buckets import GraphBucketManager
from cuda_graph.output_validation import OutputValidator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available (session-scoped)."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")


@pytest.fixture
def device(cuda_available):
    """Get the appropriate device for tests."""
    return torch.device('cuda' if cuda_available else 'cpu')


@pytest.fixture
def graph_manager(cuda_available):
    """Create a CUDA graph manager."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    return CUDAGraphManager()


@pytest.fixture
def bucket_manager(cuda_available):
    """Create a graph bucket manager."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    return GraphBucketManager(max_graphs_in_memory=3, memory_limit_mb=1024)


@pytest.fixture
def output_validator(cuda_available):
    """Create an output validator."""
    return OutputValidator(absolute_tolerance=1e-5, relative_tolerance=1e-4)


# ============================================================================
# Mock Models and Functions
# ============================================================================

@pytest.fixture
def mock_transformer_layer():
    """Create a mock transformer layer."""
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
            # Simplified transformer layer
            residual = x
            x = self.attention_norm(x)
            
            # Self-attention
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            
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
    
    return MockTransformerLayer


@pytest.fixture
def mock_llama_model():
    """Create a mock llama model."""
    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=4096, num_layers=4, vocab_size=32000):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.vocab_size = vocab_size
            
            # Create simple layers
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                for _ in range(num_layers)
            ])
            
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
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            
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
    
    return MockLlamaModel


@pytest.fixture
def simple_computation():
    """Create a simple computation function."""
    def compute(x, mask=None, pos=None, past_kv=None):
        # Simple matrix operations
        output = x @ x.transpose(-2, -1)
        output = torch.relu(output)
        output = output @ x
        return output
    
    return compute


@pytest.fixture
def complex_computation():
    """Create a more complex computation function."""
    def compute(x, mask=None, pos=None, past_kv=None):
        # Multiple operations that benefit from graph capture
        for i in range(3):
            x = x @ x.transpose(-2, -1)
            x = torch.relu(x)
            x = x @ x
            if i % 2 == 0:
                x = x + torch.ones_like(x)
        return x
    
    return compute


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_inputs(
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test inputs for transformer models.
    
    Returns:
        Tuple of (input_ids, attention_mask, position_ids)
    """
    torch.manual_seed(42)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    return input_ids, attention_mask, position_ids


def generate_random_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
    seed: int = 42
) -> torch.Tensor:
    """Generate a random tensor with reproducible randomness."""
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    if dtype in [torch.float32, torch.float64, torch.float16]:
        return torch.randn(shape, dtype=dtype, device=device)
    elif dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return torch.randint(0, 100, shape, dtype=dtype, device=device)
    elif dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=device).bool()
    else:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_tensors_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
) -> None:
    """Assert that two tensors are equal within tolerance."""
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Shape mismatch: {tensor1.shape} != {tensor2.shape}")
    
    if tensor1.dtype != tensor2.dtype:
        raise AssertionError(f"Dtype mismatch: {tensor1.dtype} != {tensor2.dtype}")
    
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
        
        error_msg = f"Tensors not equal within tolerance (rtol={rtol}, atol={atol})"
        error_msg += f"\nMax diff: {max_diff:.2e}"
        error_msg += f"\nMean diff: {mean_diff:.2e}"
        
        if msg:
            error_msg += f"\n{msg}"
        
        raise AssertionError(error_msg)


def assert_dicts_equal(
    dict1: Dict,
    dict2: Dict,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
) -> None:
    """Assert that two dictionaries are equal, handling tensors specially."""
    if set(dict1.keys()) != set(dict2.keys()):
        raise AssertionError(f"Key mismatch: {set(dict1.keys())} != {set(dict2.keys())}")
    
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            assert_tensors_equal(val1, val2, rtol=rtol, atol=atol, 
                               msg=f"Key '{key}': {msg}")
        elif isinstance(val1, dict) and isinstance(val2, dict):
            assert_dicts_equal(val1, val2, rtol=rtol, atol=atol, 
                             msg=f"Key '{key}': {msg}")
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                raise AssertionError(f"Length mismatch for key '{key}': {len(val1)} != {len(val2)}")
            
            for i, (v1, v2) in enumerate(zip(val1, val2)):
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    assert_tensors_equal(v1, v2, rtol=rtol, atol=atol,
                                       msg=f"Key '{key}[{i}]': {msg}")
                elif v1 != v2:
                    raise AssertionError(f"Value mismatch for key '{key}[{i}]': {v1} != {v2}")
        elif val1 != val2:
            raise AssertionError(f"Value mismatch for key '{key}': {val1} != {val2}")


def assert_performance_improvement(
    eager_times: List[float],
    graph_times: List[float],
    min_improvement_pct: float = 0.0,
    max_variance: float = 0.3,
    msg: str = ""
) -> Dict[str, float]:
    """
    Assert that graph execution provides performance improvement.
    
    Args:
        eager_times: List of eager execution times (ms)
        graph_times: List of graph execution times (ms)
        min_improvement_pct: Minimum improvement percentage required
        max_variance: Maximum allowed variance in improvement
        msg: Additional message for assertion errors
    
    Returns:
        Dictionary with performance statistics
    """
    import statistics
    
    # Calculate statistics
    eager_mean = statistics.mean(eager_times)
    graph_mean = statistics.mean(graph_times)
    
    eager_std = statistics.stdev(eager_times) if len(eager_times) > 1 else 0
    graph_std = statistics.stdev(graph_times) if len(graph_times) > 1 else 0
    
    # Calculate improvement
    speedup = eager_mean / graph_mean if graph_mean > 0 else 0
    improvement_pct = (speedup - 1) * 100
    
    # Calculate variance
    eager_cv = (eager_std / eager_mean) * 100 if eager_mean > 0 else 0
    graph_cv = (graph_std / graph_mean) * 100 if graph_mean > 0 else 0
    
    stats = {
        'eager_mean_ms': eager_mean,
        'graph_mean_ms': graph_mean,
        'eager_std_ms': eager_std,
        'graph_std_ms': graph_std,
        'eager_cv_pct': eager_cv,
        'graph_cv_pct': graph_cv,
        'speedup': speedup,
        'improvement_pct': improvement_pct
    }
    
    # Check minimum improvement
    if improvement_pct < min_improvement_pct:
        error_msg = f"Performance improvement below minimum: {improvement_pct:.1f}% < {min_improvement_pct:.1f}%"
        if msg:
            error_msg += f"\n{msg}"
        raise AssertionError(error_msg)
    
    # Check variance
    if graph_cv > max_variance * 100:  # Convert to percentage
        error_msg = f"Graph execution variance too high: {graph_cv:.1f}% > {max_variance * 100:.1f}%"
        if msg:
            error_msg += f"\n{msg}"
        raise AssertionError(error_msg)
    
    return stats


# ============================================================================
# Benchmarking Helpers
# ============================================================================

def benchmark_function(
    func: Callable,
    inputs: List[Any],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark a function's execution time.
    
    Returns:
        Dictionary with timing statistics
    """
    import statistics
    import time
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = func(*inputs)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        _ = func(*inputs)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    return {
        'mean_ms': statistics.mean(times),
        'stddev_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'min_ms': min(times),
        'max_ms': max(times),
        'runs': len(times),
        'times': times
    }


def compare_benchmarks(
    func1: Callable,
    func2: Callable,
    inputs: List[Any],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Compare two functions' performance.
    
    Returns:
        Dictionary with comparison results
    """
    # Benchmark both functions
    stats1 = benchmark_function(func1, inputs, warmup_iterations, benchmark_iterations, device)
    stats2 = benchmark_function(func2, inputs, warmup_iterations, benchmark_iterations, device)
    
    # Calculate comparison
    speedup = stats1['mean_ms'] / stats2['mean_ms'] if stats2['mean_ms'] > 0 else 0
    improvement_pct = (speedup - 1) * 100
    
    return {
        'func1_stats': stats1,
        'func2_stats': stats2,
        'speedup': speedup,
        'improvement_pct': improvement_pct,
        'faster_func': 'func1' if speedup < 1 else 'func2' if speedup > 1 else 'equal'
    }


# ============================================================================
# File and Directory Helpers
# ============================================================================

def create_temp_test_dir() -> Path:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="cuda_graph_test_")
    return Path(temp_dir)


def save_test_data(data: Any, filename: str, temp_dir: Optional[Path] = None) -> Path:
    """Save test data to a file."""
    if temp_dir is None:
        temp_dir = create_temp_test_dir()
    
    filepath = temp_dir / filename
    
    if isinstance(data, dict):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif isinstance(data, str):
        with open(filepath, 'w') as f:
            f.write(data)
    else:
        # Try to serialize as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    return filepath


def load_test_data(filepath: Path) -> Any:
    """Load test data from a file."""
    with open(filepath, 'r') as f:
        if filepath.suffix == '.json':
            return json.load(f)
        else:
            return f.read()


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_graph_vs_eager(
    graph_func: Callable,
    eager_func: Callable,
    inputs: List[Any],
    validator: Optional[OutputValidator] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Dict[str, Any]:
    """
    Validate that graph execution produces same results as eager execution.
    
    Returns:
        Dictionary with validation results
    """
    if validator is None:
        validator = OutputValidator(absolute_tolerance=atol, relative_tolerance=rtol)
    
    # Run both functions
    eager_output = eager_func(*inputs)
    graph_output = graph_func(*inputs)
    
    # Compare outputs
    comparison = validator._compare_outputs(eager_output, graph_output)
    
    return {
        'eager_output': eager_output,
        'graph_output': graph_output,
        'comparison': comparison,
        'passed': comparison['all_passed']
    }


# ============================================================================
# Test Decorators
# ============================================================================

def cuda_required(func):
    """Decorator to skip test if CUDA is not available."""
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def performance_test(func):
    """Decorator to mark a test as a performance test."""
    @pytest.mark.performance
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def integration_test(func):
    """Decorator to mark a test as an integration test."""
    @pytest.mark.integration
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# Test Data Classes
# ============================================================================

class TestConfig:
    """Configuration for tests."""
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 512,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_layers: int = 4,
        device: str = 'cuda'
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
    
    def generate_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test inputs based on configuration."""
        return generate_test_inputs(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=self.device
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'device': self.device
        }


# ============================================================================
# Main Test Runner Helper
# ============================================================================

def run_tests_with_report(
    test_files: List[str],
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run tests and generate a report.
    
    Args:
        test_files: List of test files to run
        output_dir: Directory to save report (None for temp directory)
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary with test results
    """
    import pytest
    import tempfile
    
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="test_report_"))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_args = test_files + [
        f"--tb=short",
        f"--junitxml={output_dir / 'test_results.xml'}",
        f"--html={output_dir / 'test_report.html'}",
        f"--self-contained-html"
    ]
    
    if verbose:
        test_args.append("-v")
    
    exit_code = pytest.main(test_args)
    
    # Collect results
    results = {
        'exit_code': exit_code,
        'output_dir': str(output_dir),
        'test_files': test_files,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Try to read JUnit XML report
    junit_file = output_dir / 'test_results.xml'
    if junit_file.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # Extract basic statistics
            for testsuite in root.findall('testsuite'):
                results.update({
                    'tests': int(testsuite.get('tests', 0)),
                    'failures': int(testsuite.get('failures', 0)),
                    'errors': int(testsuite.get('errors', 0)),
                    'skipped': int(testsuite.get('skipped', 0)),
                    'time': float(testsuite.get('time', 0))
                })
        except Exception as e:
            results['parse_error'] = str(e)
    
    return results