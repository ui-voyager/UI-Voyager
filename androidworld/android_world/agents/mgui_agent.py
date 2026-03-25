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

"""MGUI Agent using Qwen2.5-VL for mobile device interaction."""

import json
import logging
import math
import re
import time
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action

# Configure logging to display INFO level messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# try:
# from qwen_vl_utils import smart_resize
# except ImportError:
#     logging.warning("qwen_vl_utils not available. Smart resize will be skipped.")



SYSTEM_PROMPT_COT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 1080x2400.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", \\"camera\\", \\"clear\\".\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate", "answer"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=answer`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    """Process and resize image according to Qwen2.5-VL requirements.

    Args:
        image: Input image as PIL Image or dict with bytes.
        max_pixels: Maximum number of pixels allowed.
        min_pixels: Minimum number of pixels required.

    Returns:
        Processed PIL Image.
    """
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    SPATIAL_MERGE_SIZE = 2
    image_patch_size = 14
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)

    # re_h, re_w = smart_resize(width=image.width, height=image.height, factor=patch_factor)
    # image = image.resize((re_w, re_h))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class MGUIAgent(base_agent.EnvironmentInteractingAgent):
    """MGUI Agent using Qwen2.5-VL for mobile GUI interaction."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = 'MGUIAgent',
        wait_after_action_seconds: float = 2.0,
        base_screenshot_dir: str = '/apdcephfs_tj4/share_303922427/altmanyang/logs/mgui_debug_screenshots',
    ):
        """Initializes the MGUI Agent.

        Args:
            env: The environment.
            llm: The multimodal LLM wrapper.
            name: The agent name.
            wait_after_action_seconds: Seconds to wait for the screen to stabilize
                after executing an action.
            base_screenshot_dir: Base directory for saving screenshots.
        """
        super().__init__(env, name)

        self.llm = llm
        self.history = []
        self.wait_after_action_seconds = wait_after_action_seconds

        # Task-specific screenshot directory management
        self.base_screenshot_dir = base_screenshot_dir
        self.timestamp_dir = None  # Will be created on first task
        self.current_task_dir = None
        self.task_counter = 0

        logger.info("MGUI Agent initialized successfully")

    def construct_prompt(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construct the prompt for the model.

        Args:
            observation: Dictionary containing 'task', 'history', and 'image'.

        Returns:
            List of message dictionaries for the model.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT_COT}]
        messages.append({
            "role": "user",
            "content": [{'type': 'text', 'text': "The user query: " + observation['task'] + '\n\n'}]
        })

        if len(observation['history']) > 0:
            messages[-1]['content'].append({
                'type': 'text',
                'text': "Task progress (You have done the following operation on the current device):\n"
            })
            for i, his in enumerate(observation['history']):
                if '<conclusion>' in his and '</conclusion>' in his:
                    conclusion_part = his.split('<conclusion>')[1].split('</conclusion>')[0].strip()
                else:
                    conclusion_part = 'Omitted'
                messages[-1]['content'].append({
                    'type': 'text',
                    'text': f"Step {i+1}: {conclusion_part};\n"
                })

        messages[-1]['content'].append({
            'type': 'text',
            'text': "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your observation and action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."
        })

        # Process and add image
        processed_image = process_image(observation['image'], 4194304, 262144)
        messages[-1]['content'].append({'type': 'image', 'image': processed_image})

        return messages

    def parse_action_from_response(self, response: str) -> Optional[json_action.JSONAction]:
        """Parse action from model response.

        Args:
            response: Model's text response.

        Returns:
            JSONAction object or None if parsing fails.
        """
        try:
            # Extract tool_call content
            tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
            if not tool_call_match:
                logger.warning("No tool_call found in response")
                return None

            tool_call_str = tool_call_match.group(1).strip()
            action_data = json.loads(tool_call_str)

            # Map from mobile_use format to AndroidWorld JSONAction format
            args = action_data.get('arguments', {})
            action_type = args.get('action', '')

            if action_type == 'system_button':
                button = args.get('button', '')
                if button == 'Back':
                    return json_action.JSONAction(action_type='navigate_back')
                elif button == 'Home':
                    return json_action.JSONAction(action_type='navigate_home')
                elif button == 'Enter':
                    return json_action.JSONAction(action_type='keyboard_enter')
            elif action_type == 'click':
                coord = args.get('coordinate', [])
                if len(coord) == 2:
                    return json_action.JSONAction(
                        action_type='click',
                        x=coord[0],
                        y=coord[1]
                    )
            elif action_type == 'type':
                text = args.get('text', '')
                return json_action.JSONAction(
                    action_type='input_text',
                    text=text
                )
            elif action_type == 'swipe':
                coord1 = args.get('coordinate', [])
                coord2 = args.get('coordinate2', [])
                # Determine scroll direction
                if len(coord1) == 2 and len(coord2) == 2:
                    dx = coord2[0] - coord1[0]
                    dy = coord2[1] - coord1[1]
                    # scroll is reverse relative to swipe
                    if abs(dx) > abs(dy):
                        direction = 'left' if dx > 0 else 'right'
                    else:
                        direction = 'up' if dy > 0 else 'down'
                    return json_action.JSONAction(
                        action_type='scroll',
                        direction=direction
                    )
            elif action_type == 'open':
                app_name = args.get('text', '')
                return json_action.JSONAction(
                    action_type='open_app',
                    app_name=app_name
                )
            elif action_type == 'wait':
                return json_action.JSONAction(action_type='wait')
            elif action_type == 'terminate':
                status = args.get('status', 'success')
                goal_status = 'complete' if status == 'success' else 'infeasible'
                return json_action.JSONAction(
                    action_type='status',
                    goal_status=goal_status
                )
            elif action_type == 'answer':
                text = args.get('text', '')
                return json_action.JSONAction(
                    action_type='answer',
                    text=text
                )

        except Exception as e:
            logger.error("Error parsing action: %s", str(e))
            return None

        return None

    def generate_response(self, observation: Dict[str, Any]) -> str:
        """Generate response from the model.

        Args:
            observation: Current observation dictionary.

        Returns:
            Generated text response.
        """
        messages = self.construct_prompt(observation)

        # Build text prompt from messages
        text_prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                text_prompt += msg['content'] + "\n\n"
            elif msg['role'] == 'user':
                for item in msg['content']:
                    if item['type'] == 'text':
                        text_prompt += item['text']

        # Extract image from messages
        image = messages[-1]['content'][-1]['image']

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Call LLM API
        response, is_safe, raw_response = self.llm.predict_mm(
            text_prompt,
            [image]
        )

        if not raw_response:
            raise RuntimeError('Error calling LLM in generate_response.')

        return response

    def step(self, goal: str, task_name: Optional[str] = None) -> base_agent.AgentInteractionResult:
        """Performs a step of the agent on the environment.

        Args:
            goal: The goal/task description.
            task_name: Optional task name for organizing screenshots.

        Returns:
            AgentInteractionResult with done status and step data.
        """
        step_data = {
            'goal': goal,
            'raw_screenshot': None,
            'before_screenshot': None,
            'after_screenshot': None,
            'model_response': None,
            'action': None,
            'success': False,
        }

        logger.info('----------step %s----------', len(self.history) + 1)

        # Initialize timestamp directory on first task
        if self.timestamp_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.timestamp_dir = os.path.join(self.base_screenshot_dir, timestamp)
            os.makedirs(self.timestamp_dir, exist_ok=True)
            logger.info('Created timestamp directory: %s', self.timestamp_dir)

        # Initialize task directory on first step
        if self.current_task_dir is None and task_name:
            self.task_counter += 1
            self.current_task_dir = os.path.join(
                self.timestamp_dir, f'task_{self.task_counter}_{task_name}_temp'
            )
            os.makedirs(self.current_task_dir, exist_ok=True)
            logger.info('Created task directory: %s', self.current_task_dir)

        # Get current state before action
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_screenshot = state.pixels.copy()
        step_data['raw_screenshot'] = state.pixels.copy()
        step_data['before_screenshot'] = before_screenshot

        # Save screenshot with new naming scheme
        if self.current_task_dir:
            step_num = len(self.history) + 1
            screenshot_path = os.path.join(
                self.current_task_dir, f'step_{step_num}.png'
            )
            cv2.imwrite(screenshot_path, cv2.cvtColor(before_screenshot, cv2.COLOR_RGB2BGR))
            logger.info('Saved screenshot to: %s', screenshot_path)

        # Prepare observation
        screenshot_pil = Image.fromarray(before_screenshot)
        observation = {
            'task': goal,
            'history': self.history,
            'image': screenshot_pil,
        }

        # Generate response
        try:
            response = self.generate_response(observation)
            step_data['model_response'] = response
            logger.info("Model response: %s", response)

            # Save model response to text file
            if self.current_task_dir:
                step_num = len(self.history) + 1
                response_path = os.path.join(
                    self.current_task_dir, f'res_step_{step_num}.txt'
                )
                with open(response_path, 'w', encoding='utf-8') as f:
                    f.write(f'<goal>\n{goal}\n</goal>\n\n')
                    f.write(response)
                logger.info('Saved response to: %s', response_path)

            # Parse action
            action = self.parse_action_from_response(response)
            if action is None:
                logger.warning("Failed to parse action from response")
                self.history.append(response)
                return base_agent.AgentInteractionResult(False, step_data)

            step_data['action'] = action
            logger.info("Parsed action: %s", action)

            # Check if done
            if action.action_type == 'status':
                self.history.append(response)
                step_data['success'] = True
                return base_agent.AgentInteractionResult(True, step_data)

            # Execute action
            self.env.execute_action(action)
            logger.info('Action executed successfully')

            # Wait for screen to stabilize after action
            time.sleep(self.wait_after_action_seconds)

            # Get state after action
            state_after = self.env.get_state(wait_to_stabilize=False)
            after_screenshot = state_after.pixels.copy()
            step_data['after_screenshot'] = after_screenshot

            step_data['success'] = True

            # Add to history
            self.history.append(response)

            return base_agent.AgentInteractionResult(False, step_data)

        except Exception as e:
            logger.error("Error in step: %s", str(e))
            step_data['error'] = str(e)
            return base_agent.AgentInteractionResult(False, step_data)

    def reset(self, go_home: bool = False) -> None:
        """Resets the agent.

        Args:
            go_home: Whether to navigate to home screen.
        """
        super().reset(go_home)
        self.history = []
        self.current_task_dir = None  # Reset for next task
        logger.info("MGUI Agent reset")

    def finalize_task(self, success: bool) -> None:
        """Finalize task by renaming directory based on success status.

        Args:
            success: Whether the task was successful.
        """
        if self.current_task_dir and '_temp' in self.current_task_dir:
            old_dir = self.current_task_dir
            suffix = '1' if success else '0'
            new_dir = old_dir.replace('_temp', f'_{suffix}')
            try:
                os.rename(old_dir, new_dir)
                logger.info('Renamed task directory: %s -> %s', old_dir, new_dir)
                self.current_task_dir = new_dir
            except Exception as e:
                logger.error('Failed to rename task directory: %s', str(e))
