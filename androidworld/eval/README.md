# Android World Modular Evaluation Framework

A modular evaluation framework with decoupled **LLM**, **Agent**, and **Env** modules.

## Directory Structure

```
eval/
├── runner.py             # Core evaluation runner
├── agents/               # Agent implementations
│   ├── base_agent.py     # Base agent class
│   └── qwen_agent.py     # Qwen3-VL agent (vLLM)
├── clients/              # LLM clients
│   ├── base_client.py    # Base client class
│   └── openai_client.py  # OpenAI-compatible API (vLLM)
├── configs/              # YAML configurations
│   └── Qwen3-VL-4B-Instruct.yaml
└── prompts/              # System prompts
    └── qwen3vl_instruct.md
```

## Quick Start

### 1. Parallel Evaluation

```bash
# Run with 8 parallel workers, 2 repeats
./run_android_eval.sh 8 Qwen3-VL-4B-Instruct 5556 AndroidWorldAvd 2

# Monitor logs
tail -f logs/eval_*.log

# Stop evaluation
kill $(cat logs/eval.pid)
```

### 2. Run Specific Tasks

```bash
# Via command line
python run_eval_parallel.py --config eval/configs/Qwen3-VL-4B-Instruct.yaml \
    --tasks AudioRecorderRecordAudio,CameraTakePhoto

# Via task file
python run_eval_parallel.py --config eval/configs/Qwen3-VL-4B-Instruct.yaml \
    --tasks_file task_list.txt
```

## Configuration

### YAML Config Structure

```yaml
env:
  console_port: 5556
  grpc_port: 8554
  adb_path: ~/android/platform-tools/adb

llm:
  type: openai
  base_url: http://localhost:8000
  model: Qwen3-VL-4B-Instruct

agent:
  type: local
  prompt_name: qwen3vl_instruct
  sft_data_dir: ./sft_data

eval:
  tasks: null           # null = all tasks
  n_task_combinations: 1
```

### Shell Script Arguments

```bash
./run_android_eval.sh [workers] [config] [start_port] [avd_name] [repeats]

# Example: 8 workers, port 5556, 2 repeats
./run_android_eval.sh 8 Qwen3-VL-4B-Instruct 5556 AndroidWorldAvd 2
```

## Extending

### Add New LLM Client

```python
# eval/clients/my_client.py
from eval.clients.base_client import BaseLLMClient

class MyClient(BaseLLMClient):
    def chat_completion(self, messages, system_prompt=None, **kwargs):
        # Implement chat interface
        pass
    
    def multimodal_completion(self, text_prompt, images, system_prompt=None, **kwargs):
        # Implement multimodal interface
        pass
```

### Add New Agent

```python
# eval/agents/my_agent.py
from eval.agents.base_agent import BaseEvalAgent, AgentStepResult

class MyAgent(BaseEvalAgent):
    def step(self, goal, task_name=None):
        # Implement single step logic
        return AgentStepResult(done=False, data={})
```

## Output Structure

```
logs/
├── eval_*.log              # Main log
├── worker_*_repeat_*.log   # Worker logs
├── parallel_summary_repeat_*.json
└── detailed_results.csv

sft_data/
├── images/                 # All screenshots
└── repeat_XX/              # JSONL per repeat
    └── sft_data-*.jsonl
```
