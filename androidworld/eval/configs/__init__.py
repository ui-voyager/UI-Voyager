"""Configuration module - Manages evaluation configs"""

import os
import yaml
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


DEFAULT_CONFIG = {
    'env': {
        'console_port': 5556,
        'grpc_port': 8554,
        'adb_path': '~/android/platform-tools/adb',
        'emulator_path': '/root/android/emulator/emulator',
        'avd_name': 'AndroidWorldAvd',
        'android_sdk_root': '/root/android/',
        'android_avd_home': '/root/android/avd/',
    },
    'llm': {
        'type': 'openai',
        'base_url': 'http://localhost:8000',
        'model': 'Qwen3-VL-4B-Instruct',
        'temperature': 0.7,
        'max_tokens': 16384,
        'top_p': 0.8,
    },
    'agent': {
        'type': 'local',
        'name': 'Qwen3VL-Agent',
        'model_name': 'qwen3vl',
        'prompt_name': 'qwen3vl_instruct',
        'wait_after_action_seconds': 2.0,
        'use_som': False,
        'resize': None,
        'history_len': 100,
        'sft_data_dir': None,
        'n_history_image': 0,
    },
    'eval': {
        'suite_family': 'android_world',
        'tasks': None,
        'n_task_combinations': 1,
        'task_random_seed': 30,
        'checkpoint_dir': '',
        'output_path': '~/android_world/runs',
    },
}


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configurations (recursive)"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
