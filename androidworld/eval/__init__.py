"""Evaluation framework core module

This module provides:
1. Evaluation runner
2. Result processing
3. Checkpoint management
"""

from eval.runner import EvalRunner
from eval.configs import load_config, get_default_config, merge_configs

__all__ = [
    'EvalRunner',
    'load_config',
    'get_default_config', 
    'merge_configs',
]
