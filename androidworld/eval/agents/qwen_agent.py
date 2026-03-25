"""Qwen Agent - Qwen VL Model Agent Implementation"""

import json
import os
import random
import re
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from absl import logging

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action
from eval.agents.base_agent import BaseEvalAgent, AgentStepResult
from eval.clients.base_client import BaseLLMClient
from eval.prompts import get_prompt


class QwenAgent(BaseEvalAgent):
    """Qwen VL Agent - Supports Qwen3-VL multimodal models"""
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm_client: BaseLLMClient,
        name: str = 'QwenAgent',
        model_name: str = 'qwen3vl',
        system_prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
        wait_after_action_seconds: float = 2.0,
        use_som: bool = False,
        resize: Optional[List[int]] = None,
        history_len: int = 100,
        sft_data_dir: Optional[str] = None,
        n_history_image: int = 0,
        repeat_id: int = 0,
        **kwargs,
    ):
        super().__init__(env, name, transition_pause=1.0)
        
        self.llm_client = llm_client
        self.model_name = model_name
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = get_prompt(prompt_name)
        self.wait_after_action_seconds = wait_after_action_seconds
        self.use_som = use_som
        self.resize = resize
        self.history_len = history_len
        self.sft_data_dir = sft_data_dir
        self.n_history_image = n_history_image
        self.repeat_id = repeat_id
        
        self._history = []
        self._original_size = None
        
        self._json_list = []
        self._history_images_paths = []
        self._history_images = []
        
        if sft_data_dir:
            os.makedirs(sft_data_dir, exist_ok=True)
            self._current_task_name = None
            self._current_task_images_dir = None
            logging.info(f'SFT data dir: {sft_data_dir}, repeat_id: {repeat_id}')
        
        logging.info(f'QwenAgent initialized: {model_name}')
    
    def reset(self, go_home: bool = False) -> None:
        """Reset agent state"""
        super().reset(go_home)
        self._history = []
        self._original_size = None
        self._json_list = []
        self._history_images_paths = []
        self._history_images = []
    
    def step(self, goal: str, task_name: Optional[str] = None) -> AgentStepResult:
        """Execute one step"""
        step_data = {
            'goal': goal,
            'before_screenshot': None,
            'after_screenshot': None,
            'model_response': None,
            'action': None,
            'success': False,
        }
        
        logging.info(f'---------- Step {len(self._history) + 1} ----------')
        
        state = self.get_post_transition_state()
        screen_width = state.pixels.shape[1]
        screen_height = state.pixels.shape[0]
        
        if self._original_size is None:
            self._original_size = (screen_width, screen_height)
        
        before_screenshot = state.pixels.copy()
        step_data['before_screenshot'] = before_screenshot
        
        screenshot_pil = Image.fromarray(before_screenshot)
        if self.resize:
            screenshot_pil = screenshot_pil.resize(self.resize)
        processed_image = np.array(screenshot_pil)
        
        text_prompt = self._construct_prompt(goal)
        
        try:
            response, raw_response = self.llm_client.predict_mm(
                text_prompt=text_prompt,
                images=[processed_image],
                system_prompt=self.system_prompt,
            )
            
            step_data['model_response'] = response
            logging.info(f'Model response: {response}')
            
            self._save_sft_data(
                goal=goal,
                task_name=task_name,
                response=response,
                text_prompt=text_prompt,
                image=processed_image,
            )
            
            action = self._parse_action(response, screen_width, screen_height)
            if action is None:
                logging.warning('Failed to parse action')
                return AgentStepResult(done=False, data=step_data)
            
            step_data['action'] = action
            logging.info(f'Parsed action: {action}')
            
            action_description = self._get_action_description(response)
            
            if action.action_type == 'status':
                self._history.append(action_description)
                step_data['success'] = True
                return AgentStepResult(done=True, data=step_data)
            
            self.env.execute_action(action)
            logging.info('Action executed')
            
            time.sleep(self.wait_after_action_seconds)
            
            state_after = self.env.get_state(wait_to_stabilize=False)
            step_data['after_screenshot'] = state_after.pixels.copy()
            step_data['success'] = True
            
            if action.action_type == 'answer':
                self._history.append(action_description)
                return AgentStepResult(done=True, data=step_data)
            
            self._history.append(action_description)
            
            return AgentStepResult(done=False, data=step_data)
            
        except Exception as e:
            logging.error(f'Execution error: {e}', exc_info=True)
            step_data['error'] = str(e)
            return AgentStepResult(done=False, data=step_data)
    
    def finalize_task(self, success: bool) -> None:
        """Save SFT data when task ends (per-task JSONL with succ/fail suffix)"""
        if not self.sft_data_dir or not self._json_list:
            return
        
        for json_data in self._json_list:
            json_data['is_success'] = success
        
        # Determine task_name from recorded data
        task_safe_name = self._json_list[0].get('task_name', 'unknown')
        
        # Build path: {sft_data_dir}/{task_name}/repeat_XX_succ.jsonl
        task_dir = os.path.join(self.sft_data_dir, task_safe_name)
        os.makedirs(task_dir, exist_ok=True)
        
        suffix = 'succ' if success else 'fail'
        jsonl_filename = f'repeat_{self.repeat_id:02d}_{suffix}.jsonl'
        jsonl_path = os.path.join(task_dir, jsonl_filename)
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for json_data in self._json_list:
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        
        logging.info(f'SFT data saved: {jsonl_path} ({len(self._json_list)} steps, success={success})')
    
    def _construct_prompt(self, goal: str) -> str:
        """Construct prompt"""
        prompt_parts = []
        
        # Task Instruction
        prompt_parts.append(f"The user query: {goal}")
        prompt_parts.append("")
        
        # Task Progress (action history)
        recent_history = self._history[-self.history_len:] if len(self._history) > self.history_len else self._history
        
        if recent_history:
            prompt_parts.append("Task progress (You have done the following operations on the current device):")
            start_step = len(self._history) - len(recent_history) + 1
            
            for i, action_text in enumerate(recent_history):
                cleaned_text = self._clean_action_text(action_text)
                prompt_parts.append(f"Step{start_step + i}: {cleaned_text}")
        
        # Current Screenshot
        prompt_parts.append("")
        prompt_parts.append("Current Screenshot: <image>")
        prompt_parts.append("")
        prompt_parts.append("Please analyze the current screenshot and history to generate the next step.")
        
        return "\n".join(prompt_parts)
    
    def _clean_action_text(self, text: str) -> str:
        """Clean special characters from action text"""
        if not text:
            return text
        
        text = text.replace('\\n', ' ')
        text = text.replace('\\r', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\', '')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        return text
    
    def _parse_action(
        self, 
        response: str, 
        screen_width: int, 
        screen_height: int
    ) -> Optional[json_action.JSONAction]:
        """Parse model response to action"""
        try:
            action_str = self._extract_tag_content(response, 'tool_call')
            if not action_str:
                logging.warning('tool_call not found')
                return json_action.JSONAction(action_type='wait')
            
            try:
                action_dict = json.loads(action_str)
            except json.JSONDecodeError as e:
                logging.error(f'JSON parse failed: {e}')
                return json_action.JSONAction(action_type='wait')
            
            action_type = action_dict.get('arguments', {}).get('action', '')
            if not action_type:
                return json_action.JSONAction(action_type='wait')
            
            x, y, x_, y_, text, direction, goal_status, app_name = (
                None, None, None, None, None, None, None, None
            )
            
            if action_type in ['click', 'long_press']:
                coord = action_dict['arguments'].get('coordinate')
                if not coord or len(coord) != 2:
                    return json_action.JSONAction(action_type='wait')
                x, y = coord
                if 'qwen3' in self.model_name.lower():
                    x = round(x / 999 * screen_width)
                    y = round(y / 999 * screen_height)
            
            elif action_type == 'swipe':
                coord1 = action_dict['arguments'].get('coordinate')
                coord2 = action_dict['arguments'].get('coordinate2')
                if not coord1 or not coord2:
                    return json_action.JSONAction(action_type='wait')
                x1, y1 = coord1
                x2, y2 = coord2
                
                if 'qwen3' in self.model_name.lower():
                    x = round(x1 / 999 * screen_width)
                    y = round(y1 / 999 * screen_height)
                    x_ = round(x2 / 999 * screen_width)
                    y_ = round(y2 / 999 * screen_height)
                else:
                    x, y = x1, y1
                    x_, y_ = x2, y2
            
            elif action_type in ['open', 'open_app']:
                action_type = 'open_app'
                app_name = action_dict['arguments'].get('text', '')
            
            elif action_type in ['type', 'input_text']:
                action_type = 'input_text'
                text = action_dict['arguments'].get('text', '')
            
            elif action_type == 'system_button':
                button = action_dict['arguments'].get('button', '')
                button_map = {
                    'Back': 'navigate_back',
                    'Home': 'navigate_home',
                    'Enter': 'keyboard_enter',
                }
                action_type = button_map.get(button, 'wait')
            
            elif action_type == 'terminate':
                action_type = 'status'
                goal_status = action_dict['arguments'].get('status', 'success')
            
            elif action_type == 'answer':
                text = action_dict['arguments'].get('text', '')
                self.env.interaction_cache = text
            
            return json_action.JSONAction(
                action_type=action_type,
                direction=direction,
                x=x,
                y=y,
                x_=x_,
                y_=y_,
                text=text,
                goal_status=goal_status,
                app_name=app_name,
            )
            
        except Exception as e:
            logging.error(f'Parse action error: {e}', exc_info=True)
            return json_action.JSONAction(action_type='wait')
    
    def _extract_tag_content(self, response: str, tag: str) -> str:
        """Extract XML tag content"""
        if f'<{tag}>' not in response:
            return ''
        
        later_half = response.split(f'<{tag}>')[1].strip('\n')
        if f'</{tag}>' in later_half:
            content = later_half.split(f'</{tag}>')[0].strip('\n')
        else:
            content = later_half.split('\n')[0]
        return content
    
    def _get_action_description(self, response: str) -> str:
        """Get action description"""
        action = self._extract_tag_content(response, 'action')
        if action:
            return action

        conclusion = self._extract_tag_content(response, 'conclusion')
        if conclusion:
            return conclusion
        
        if 'Action:' in response:
            action_start = response.find('Action:') + len('Action:')
            action_end = response.find('\n', action_start)
            if action_end == -1:
                action_end = len(response)
            action = response[action_start:action_end].strip()
            if action:
                return action
        
        return response[:200]
    
    def _save_sft_data(
        self,
        goal: str,
        task_name: Optional[str],
        response: str,
        text_prompt: str,
        image: np.ndarray,
    ) -> None:
        """Save SFT training data (per-task directory structure)"""
        if not self.sft_data_dir:
            return
        
        try:
            task_safe_name = task_name.replace('/', '_').replace('\\', '_') if task_name else 'unknown'
            
            # Lazily create per-task images directory
            if self._current_task_name != task_safe_name:
                self._current_task_name = task_safe_name
                self._current_task_images_dir = os.path.join(
                    self.sft_data_dir, task_safe_name, 'images'
                )
                os.makedirs(self._current_task_images_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            image_filename = f'repeat{self.repeat_id:02d}-step{len(self._history)}-{timestamp}-{image_id}.png'
            image_path = os.path.join(self._current_task_images_dir, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            self._history_images_paths.append(image_path)
            if self.n_history_image > 0:
                self._history_images.append(image)
            
            random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            json_data = {
                'task_id': f'{timestamp}-{task_safe_name}-{random_id}',
                'task_name': task_safe_name,
                'index': len(self._history),
                'conversations': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': text_prompt},
                    {'role': 'assistant', 'content': response}
                ],
                'images': self._history_images_paths[-(self.n_history_image + 1):]
            }
            self._json_list.append(json_data)
            
            logging.info(f'SFT data recorded: step={len(self._history)}, image={image_filename}')
            
        except Exception as e:
            logging.error(f'Failed to save SFT data: {e}')
