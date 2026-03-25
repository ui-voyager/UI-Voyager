"""Evaluation Runner - Core evaluation logic"""

import datetime
import os
import random
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type

from absl import logging
import pandas as pd

from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world.env import env_launcher
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.miniwob import miniwob_base

from eval.agents.base_agent import BaseEvalAgent


class EvalRunner:
    """Evaluation Runner
    
    Responsible for:
    1. Environment initialization
    2. Task suite creation
    3. Evaluation execution
    4. Result collection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation runner
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.env: Optional[interface.AsyncEnv] = None
        self.agent: Optional[BaseEvalAgent] = None
        self._results: List[Dict[str, Any]] = []
        self.worker_id: Optional[int] = config.get('worker_id')
    
    def setup_env(self) -> interface.AsyncEnv:
        """Initialize Android environment"""
        env_config = self.config.get('env', {})
        
        self.env = env_launcher.load_and_setup_env(
            console_port=env_config.get('console_port', 5556),
            emulator_setup=False,
            freeze_datetime=True,
            adb_path=os.path.expanduser(env_config.get('adb_path', '~/android/platform-tools/adb')),
            grpc_port=env_config.get('grpc_port', 8554),
            emulator_path=env_config.get('emulator_path', '/root/android/emulator/emulator'),
            avd_name=env_config.get('avd_name', 'AndroidWorldAvd'),
            android_sdk_root=env_config.get('android_sdk_root', '/root/android/'),
            android_avd_home=env_config.get('android_avd_home', '/root/android/avd/'),
            adb_server_port=env_config.get('adb_server_port', 5037),
        )
        
        logging.info('Android environment initialized')
        return self.env
    
    def setup_agent(self, llm_client, repeat_id: int = 0) -> BaseEvalAgent:
        """Initialize Agent
        
        Args:
            llm_client: LLM client instance
            repeat_id: Current evaluation round ID (for SFT data directory)
        
        Returns:
            Agent instance
        """
        from eval.agents import get_agent
        
        agent_config = self.config.get('agent', {})
        self.agent = get_agent(agent_config, self.env, llm_client, repeat_id=repeat_id)
        
        logging.info(f'Agent initialized: {self.agent.name}, repeat_id: {repeat_id}')
        return self.agent
    
    def create_suite(self) -> 'Suite':
        """Create task suite"""
        from android_world.suite_utils import Suite, create_suite
        
        eval_config = self.config.get('eval', {})
        suite_family = eval_config.get('suite_family', 'android_world')
        
        task_registry = registry.TaskRegistry()
        reg = task_registry.get_registry(family=suite_family)
        
        task_seed = eval_config.get('task_random_seed')

        suite_kwargs = {
            'task_registry': reg,
            'n_task_combinations': eval_config.get('n_task_combinations', 1),
            'tasks': eval_config.get('tasks'),
            'use_identical_params': False,
            'env': self.env,
        }
        if task_seed is not None:
            suite_kwargs['seed'] = task_seed
        
        suite = create_suite(**suite_kwargs)
        suite.suite_family = suite_family
        
        logging.info(f'Task suite created: {len(suite)} tasks, seed: {task_seed}')
        return suite
    
    def run(
        self,
        suite: 'Suite',
        checkpointer: Optional[checkpointer_lib.Checkpointer] = None,
        demo_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run evaluation
        
        Args:
            suite: Task suite
            checkpointer: Checkpoint manager
            demo_mode: Demo mode
        
        Returns:
            List of evaluation results
        """
        if checkpointer is None:
            eval_config = self.config.get('eval', {})
            checkpoint_dir = eval_config.get('checkpoint_dir', '')
            if not checkpoint_dir:
                output_path = os.path.expanduser(
                    eval_config.get('output_path', '~/android_world/runs')
                )
                checkpoint_dir = checkpointer_lib.create_run_directory(output_path)
            checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
        
        logging.info(f'Starting evaluation, checkpoint dir: {checkpointer.directory}')
        
        results = self._run_suite(suite, checkpointer, demo_mode)
        self._results = results
        
        return results
    
    def _run_suite(
        self,
        suite: 'Suite',
        checkpointer: checkpointer_lib.Checkpointer,
        demo_mode: bool,
    ) -> List[Dict[str, Any]]:
        """Run task suite"""
        metadata_fields = [
            constants.EpisodeConstants.GOAL,
            constants.EpisodeConstants.TASK_TEMPLATE,
            constants.EpisodeConstants.INSTANCE_ID,
            constants.EpisodeConstants.IS_SUCCESSFUL,
            constants.EpisodeConstants.EPISODE_LENGTH,
            constants.EpisodeConstants.RUN_TIME,
            constants.EpisodeConstants.EXCEPTION_INFO,
            constants.EpisodeConstants.AUX_DATA,
        ]
        
        completed_tasks, failed_tasks = self._get_task_info(
            checkpointer.load(fields=metadata_fields)
        )
        
        results: List[Dict[str, Any]] = []
        correct, total = 0, 0
        
        for name, instances in suite.items():
            self._log_and_print(f'Running task: {name}\n{"=" * 50}')
            
            for i, instance in enumerate(instances):
                instance_name = f'{instance.name}{checkpointer_lib.INSTANCE_SEPARATOR}{i}'
                
                if instance_name in completed_tasks:
                    results.extend(completed_tasks[instance_name])
                    continue
                if instance_name in failed_tasks:
                    results.extend(failed_tasks[instance_name])
                    continue
                
                episode = self._run_task(instance, demo_mode)
                episode[constants.EpisodeConstants.AGENT_NAME] = self.agent.name
                episode[constants.EpisodeConstants.INSTANCE_ID] = i
                
                checkpointer.save_episodes([episode], instance_name)
                results.append({k: episode[k] for k in metadata_fields})
                
                if episode[constants.EpisodeConstants.EXCEPTION_INFO] is None:
                    correct += episode[constants.EpisodeConstants.IS_SUCCESSFUL]
                    total += 1
                    
                    success_rate = correct / total * 100
                    self._log_and_print(f'Current success rate: {correct}/{total} ({success_rate:.1f}%)')
        
        return results
    
    def _run_task(
        self,
        task: task_eval.TaskEval,
        demo_mode: bool,
    ) -> Dict[str, Any]:
        """Run single task"""
        start = time.time()

        try:
            if task.start_on_home_screen:
                self.agent.reset(go_home=True)
            else:
                self.agent.reset(go_home=False)

            eval_config = self.config.get('eval', {})
            task_seed = eval_config.get('task_random_seed')
            
            if task_seed is not None:
                random.seed(task_seed)
                self._log_and_print(f'Task {task.name}: random seed set to {task_seed}')

            task.initialize_task(self.env)
            self._log_and_print(f'Running task {task.name}, goal: "{task.goal}"')
            
            interaction_results = self._run_episode(task)
            
            task_successful = task.is_successful(self.env)
            agent_successful = task_successful if interaction_results.done else 0.0
            
            if hasattr(self.agent, 'finalize_task'):
                self.agent.finalize_task(agent_successful > 0.5)
            
            success_msg = 'Task succeeded ✅' if agent_successful > 0.5 else 'Task failed ❌'
            self._log_and_print(f'{success_msg}: {task.goal}')
            
            result = {
                constants.EpisodeConstants.GOAL: task.goal,
                constants.EpisodeConstants.TASK_TEMPLATE: task.name,
                constants.EpisodeConstants.EPISODE_DATA: interaction_results.step_data,
                constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
                constants.EpisodeConstants.RUN_TIME: time.time() - start,
                constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
                constants.EpisodeConstants.EPISODE_LENGTH: len(
                    interaction_results.step_data.get(constants.STEP_NUMBER, [])
                ),
                constants.EpisodeConstants.AUX_DATA: interaction_results.aux_data,
                constants.EpisodeConstants.EXCEPTION_INFO: None,
                constants.EpisodeConstants.SEED: task.params.get(
                    constants.EpisodeConstants.SEED
                ),
            }
            
            task.tear_down(self.env)
            return result
            
        except Exception as e:
            self._log_and_print(f'Task error, skipping {task.name}: {e}')
            traceback.print_exc()
            return self._create_failed_result(
                task.name, task.goal, traceback.format_exc(), time.time() - start
            )
    
    def _run_episode(self, task: task_eval.TaskEval) -> episode_runner.EpisodeResult:
        """Run an episode"""
        max_n_steps = int(10 * task.complexity) if task.complexity else 10

        print(f'Task {task.name}: complexity={task.complexity}, max_n_steps={max_n_steps}')

        termination_fn = None
        if task.name.lower().startswith('miniwob'):
            termination_fn = miniwob_base.is_episode_terminated
        
        return episode_runner.run_episode(
            goal=task.goal,
            agent=self.agent,
            max_n_steps=max_n_steps,
            start_on_home_screen=False,
            termination_fn=termination_fn,
            task_name=task.name,
            worker_id=self.worker_id,
        )
    
    def _get_task_info(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> tuple:
        """Get completed and failed task info"""
        import collections
        
        completed = collections.defaultdict(list)
        failed = collections.defaultdict(list)
        
        for episode in episodes:
            instance_name = (
                episode[constants.EpisodeConstants.TASK_TEMPLATE]
                + checkpointer_lib.INSTANCE_SEPARATOR
                + str(episode[constants.EpisodeConstants.INSTANCE_ID])
            )
            if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
                failed[instance_name].append(episode)
            else:
                completed[instance_name].append(episode)
        
        return completed, failed
    
    def _create_failed_result(
        self,
        name: str,
        goal: str,
        exception: str,
        run_time: float,
    ) -> Dict[str, Any]:
        """Create failed result"""
        import numpy as np
        
        return {
            constants.EpisodeConstants.GOAL: goal,
            constants.EpisodeConstants.TASK_TEMPLATE: name,
            constants.EpisodeConstants.EPISODE_DATA: np.nan,
            constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
            constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
            constants.EpisodeConstants.RUN_TIME: run_time,
            constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
            constants.EpisodeConstants.EXCEPTION_INFO: exception,
            constants.EpisodeConstants.AUX_DATA: None,
        }
    
    def _log_and_print(self, msg: str) -> None:
        """Log and print"""
        logging.info(msg)
        print(msg)
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get results summary"""
        from android_world.suite_utils import process_episodes
        return process_episodes(self._results, print_summary=True)
    
    def close(self) -> None:
        """Close resources"""
        if self.env:
            self.env.close()
            logging.info('Environment closed')
