"""Prompts module - Manages system prompts"""

import os
from typing import Optional

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_NAME = 'qwen3vl_instruct'


def load_prompt(name: str) -> str:
    """Load prompt by name
    
    Args:
        name: Prompt name (without .md suffix)
    
    Returns:
        Prompt content
    """
    filepath = os.path.join(_PROMPTS_DIR, f'{name}.md')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Prompt file not found: {filepath}')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    if lines and lines[0].startswith('# '):
        content = '\n'.join(lines[1:])
    
    return content.strip()


def get_prompt(name: Optional[str] = None) -> str:
    """Get system prompt by name
    
    Args:
        name: Prompt name (without .md suffix), None for default
    
    Returns:
        Prompt content
    """
    prompt_name = name if name else DEFAULT_PROMPT_NAME
    return load_prompt(prompt_name)


def list_available_prompts() -> list:
    """List all available prompts"""
    prompts = []
    for f in os.listdir(_PROMPTS_DIR):
        if f.endswith('.md'):
            prompts.append(f[:-3])
    return prompts
