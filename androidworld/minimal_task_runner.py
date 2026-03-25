# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import os
import random
from typing import Type

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer
from android_world.agents import t3a
from android_world.agents import m3a
from android_world.agents import mgui_agent, phone_agent, qwen_agent
from android_world.env import env_launcher
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
      os.path.expanduser('~/android/platform-tools/adb')
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5556,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)


def _main() -> None:
  """Runs a single task."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  # env.reset(go_home=True)
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  if _TASK.value:
    if _TASK.value not in aw_registry:
      raise ValueError('Task {} not found in registry.'.format(_TASK.value))
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
  else:
    task_type: Type[task_eval.TaskEval] = random.choice(
        list(aw_registry.values())
    )
  params = task_type.generate_random_params()
  task = task_type(params)
  task.initialize_task(env)
  # agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # agent = mgui_agent.MGUIAgent(
  #   env, 
  #   infer.MGUIWrapper('/apdcephfs_tj4/share_303922427/altmanyang/logs/GUI-RL/qwenvl_verl_7b_11_13_2025_1823/global_step_150/actor/huggingface')
  # )
  # agent = phone_agent.PhoneAgent(env, infer.PhoneAgentWrapper('autoglm-phone-9b'))
  agent = qwen_agent.QwenAgent(
    env,
    infer.OAIWrapper(
      model_name="",
      endpoint='http://127.0.0.1:22002/v1/chat/completions',
      temperature=0.7,
      top_p=0.8,
      top_k=20,
      seed=3407,
      repetition_penalty=1,
      presence_penalty=1.5,
      max_tokens=1000,
      greedy=True,
    ),
    model_name='qwen3vl',
    save_logs=True,
    base_screenshot_dir='/apdcephfs_tj4/share_303922427/altmanyang/logs/qwen3vl_235b_agent_screenshots'
  )

  print('Goal: ' + str(task.goal))
  is_done = False
  for _ in range(int(task.complexity * 10)):
    response = agent.step(task.goal, task.name)
    if response.done:
      is_done = True
      break
  agent_successful = is_done and task.is_successful(env) == 1
  print("is_done: ", is_done)
  print("task.is_successful(env): ", task.is_successful(env))
  print("task.is_successful(env) == 1: ", task.is_successful(env) == 1)
  print("agent_successful: ", agent_successful)
  print(
      f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
      f' {task.goal}'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
