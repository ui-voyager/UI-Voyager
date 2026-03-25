"""Agent module - Provides Agent implementations for evaluation"""

from eval.agents.base_agent import BaseEvalAgent
from eval.agents.qwen_agent import QwenAgent

__all__ = [
    'BaseEvalAgent',
    'QwenAgent',
]


def get_agent(config: dict, env, llm_client, repeat_id: int = 0) -> BaseEvalAgent:
    """Create Agent based on configuration
    
    Args:
        config: Agent configuration dictionary
        env: Android environment
        llm_client: LLM client
        repeat_id: Current evaluation round ID (for SFT data directory)
    
    Returns:
        BaseEvalAgent instance
    """
    agent_type = config.get('type', 'qwen')
    
    if agent_type in ['qwen', 'local']:
        return QwenAgent(
            env=env,
            llm_client=llm_client,
            name=config.get('name', 'QwenAgent'),
            model_name=config.get('model_name', 'qwen3vl'),
            system_prompt=config.get('system_prompt'),
            prompt_name=config.get('prompt_name'),
            wait_after_action_seconds=config.get('wait_after_action_seconds', 2.0),
            use_som=config.get('use_som', False),
            resize=config.get('resize'),
            history_len=config.get('history_len', 100),
            sft_data_dir=config.get('sft_data_dir'),
            n_history_image=config.get('n_history_image', 0),
            repeat_id=repeat_id,
        )
    else:
        raise ValueError(f"Unsupported Agent type: {agent_type}")
