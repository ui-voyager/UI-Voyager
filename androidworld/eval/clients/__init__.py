"""LLM Client module - Supports OpenAI-compatible API (vLLM local deployment)"""

from eval.clients.base_client import BaseLLMClient
from eval.clients.openai_client import OpenAIClient

__all__ = [
    'BaseLLMClient',
    'OpenAIClient',
]


def get_llm_client(config: dict) -> BaseLLMClient:
    """Create LLM client based on configuration
    
    Args:
        config: LLM configuration dictionary, must contain 'type' field
    
    Returns:
        BaseLLMClient instance
    
    Raises:
        ValueError: Unsupported LLM type
    """
    llm_type = config.get('type', 'openai')
    
    if llm_type == 'openai':
        return OpenAIClient(
            base_url=config.get('base_url', 'http://localhost:8000'),
            model=config.get('model', 'Qwen3-VL-4B-Instruct'),
            api_key=config.get('api_key', 'EMPTY'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 16384),
            top_p=config.get('top_p', 0.8),
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
