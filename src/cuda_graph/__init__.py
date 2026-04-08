"""
CUDA Graph Compilation Module for PipeLLM.

This module implements static CUDA graph capture for LLM decode loops
to eliminate per-token dispatch overhead (Phase 1, Task 1.3).
"""

__version__ = "0.1.0"
__author__ = "PipeLLM Team"