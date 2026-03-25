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

"""Utilies for actuation."""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import json_action
from android_world.env import representation_utils


@dataclass
class ADBActionResult:
  """Stores action execution results and error information.
  
  Can be passed back to qwen_agent for decision-making in the future.
  """
  success: bool = True
  action: Optional[json_action.JSONAction] = None
  adb_errors: list[dict] = field(default_factory=list)
  
  def add_adb_error(self, action_desc: str, status: str, output: str = '', error_message: str = ''):
    """Add an ADB command execution error."""
    self.success = False
    self.adb_errors.append({
      'action_desc': action_desc,
      'status': status,
      'output': output,
      'error_message': error_message,
    })
  
  def log_errors(self):
    """Print all collected error messages."""
    if self.adb_errors:
      logging.error('=== ADB Errors Summary ===')
      logging.error(f'Action: {self.action}')
      for i, err in enumerate(self.adb_errors, 1):
        logging.error(f'[ADB Error {i}] Command: {err["action_desc"]}')
        logging.error(f'  Status: {err["status"]}')
        if err['output']:
          logging.error(f'  Output: {err["output"]}')
        if err['error_message']:
          logging.error(f'  Error Message: {err["error_message"]}')
  
  def to_dict(self) -> dict:
    """Convert to dict for serialization and passing to agent."""
    return {
      'success': self.success,
      'action': str(self.action) if self.action else None,
      'adb_errors': self.adb_errors,
    }


def _check_adb_response(
    response: adb_pb2.AdbResponse,
    action_desc: str,
    result: ADBActionResult,
) -> bool:
  """Check ADB response status and log details to ADBActionResult.
  
  Args:
      response: ADB command response object
      action_desc: Action description for logging
      result: ActionResult object for collecting error information
      
  Returns:
      True if the response status is OK, otherwise False
  """
  if response.status != adb_pb2.AdbResponse.Status.OK:
    status_str = str(response.status)
    output = ''
    error_message = ''
    
    try:
      output = response.generic.output.decode('utf-8', errors='replace') if response.generic.output else ''
    except Exception as e:
      output = f'Failed to decode: {e}'
    
    if hasattr(response, 'error_message') and response.error_message:
      error_message = response.error_message
    
    result.add_adb_error(action_desc, status_str, output, error_message)
    return False
  return True


def execute_adb_action(
    action: json_action.JSONAction,
    screen_elements: list[Any],  # list[UIElement]
    screen_size: tuple[int, int],
    env: env_interface.AndroidEnvInterface,
) -> ADBActionResult:
  """Execute an action based on a JSONAction object.

  Args:
      action: JSONAction object containing the action to be executed.
      screen_elements: List of UI elements on the screen.
      screen_size: The (width, height) of the screen.
      env: The environment to execute the action in.
      
  Returns:
      ADBActionResult containing execution results and error information.
  """
  result = ADBActionResult(action=action)
  
  try:
    _execute_adb_action_impl(action, screen_elements, screen_size, env, result)
  except Exception as e:
    # Print exception info directly
    result.success = False
    print(f'Invalid action due to exception: {type(e).__name__}: {e}, action: {action}')
  
  # Print all collected error messages
  if not result.success:
    result.log_errors()
  
  return result


def _execute_adb_action_impl(
    action: json_action.JSONAction,
    screen_elements: list[Any],  # list[UIElement]
    screen_size: tuple[int, int],
    env: env_interface.AndroidEnvInterface,
    result: ADBActionResult,
) -> None:
  """Internal implementation of execute_adb_action.
  
  Args:
      action: JSONAction object containing the action to be executed.
      screen_elements: List of UI elements on the screen.
      screen_size: The (width, height) of the screen.
      env: The environment to execute the action in.
      result: ADBActionResult object for collecting error information.
  """
  if action.action_type in ['click', 'double_tap', 'long_press']:
    idx = action.index
    x = action.x
    y = action.y
    if idx is not None:
      if idx < 0 or idx >= len(screen_elements):
        raise ValueError(
            f'Invalid element index: {idx}, must be between 0 and'
            f' {len(screen_elements)-1}.'
        )
      element = screen_elements[idx]
      if element.bbox_pixels is None:
        raise ValueError('Bbox is not present on element.')
      x, y = element.bbox_pixels.center
      x, y = int(x), int(y)
      if action.action_type == 'click':
        response = adb_utils.tap_screen(x, y, env)
        _check_adb_response(response, f'tap_screen({x}, {y})', result)
      elif action.action_type == 'double_tap':
        response = adb_utils.double_tap(x, y, env)
        _check_adb_response(response, f'double_tap({x}, {y})', result)
      else:
        response = adb_utils.long_press(x, y, env)
        _check_adb_response(response, f'long_press({x}, {y})', result)
    elif x is not None and y is not None:
      x, y = int(x), int(y)
      if action.action_type == 'click':
        response = adb_utils.tap_screen(x, y, env)
        _check_adb_response(response, f'tap_screen({x}, {y})', result)
      elif action.action_type == 'double_tap':
        response = adb_utils.double_tap(x, y, env)
        _check_adb_response(response, f'double_tap({x}, {y})', result)
      else:
        response = adb_utils.long_press(x, y, env)
        _check_adb_response(response, f'long_press({x}, {y})', result)
    else:
      raise ValueError(f'Invalid click action: {action}')

  elif action.action_type == 'input_text':
    text = action.text
    if text:
      if action.index is not None or (
          action.x is not None and action.y is not None
      ):
        # First focus on enter text UI element.
        click_action = copy.deepcopy(action)
        click_action.action_type = 'click'
        # Pass result in recursive call to collect errors
        _execute_adb_action_impl(click_action, screen_elements, screen_size, env, result)
        time.sleep(1.0)

      if action.clear_text:
        # Select all existing text and delete it.
        response = adb_utils.issue_generic_request(
            [
                'shell',
                'input',
                'keycombination',
                '113',
                '29',
                '&&',
                'input',
                'keyevent',
                '67',
            ],
            env,
        )
        _check_adb_response(response, 'clear_text (keycombination)', result)
        time.sleep(1.0)

      response = adb_utils.type_text(text, env, timeout_sec=10)
      _check_adb_response(response, f'type_text({repr(text)})', result)
      response = adb_utils.press_enter_button(env)
      _check_adb_response(response, 'press_enter_button', result)
    else:
      logging.warning(
          'Input_text action indicated, but no text provided. No '
          'action will be executed.'
      )

  elif action.action_type == 'keyboard_enter':
    response = adb_utils.press_enter_button(env)
    _check_adb_response(response, 'press_enter_button', result)

  elif action.action_type == 'navigate_home':
    response = adb_utils.press_home_button(env)
    _check_adb_response(response, 'press_home_button', result)

  elif action.action_type == 'navigate_back':
    response = adb_utils.press_back_button(env)
    _check_adb_response(response, 'press_back_button', result)

  elif action.action_type == 'press_keyboard':
    response = adb_utils.press_keyboard_generic(action.keycode, env)
    _check_adb_response(response, f'press_keyboard_generic({action.keycode})', result)
  elif action.action_type == 'drag_and_drop':
    if action.touch_xy is not None and action.lift_xy is not None:
      command = adb_utils.generate_drag_and_drop_command(
          action.touch_xy[0],
          action.touch_xy[1],
          action.lift_xy[0],
          action.lift_xy[1],
          4000,
      )
      response = adb_utils.issue_generic_request(command, env)
      _check_adb_response(response, f'drag_and_drop({action.touch_xy} -> {action.lift_xy})', result)
    else:
      logging.warning(
          'Drag and drop action indicated, but no coordinates provided. No '
          'action will be executed.'
      )
  elif action.action_type == 'scroll':

    screen_width, screen_height = screen_size
    if action.index:
      x_min, y_min, x_max, y_max = (
          max(screen_elements[action.index].bbox_pixels.x_min, 0),
          max(screen_elements[action.index].bbox_pixels.y_min, 0),
          min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
          min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
      )
    else:
      x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

    start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    direction = action.direction
    if direction == 'down':
      end_x, end_y = (x_min + x_max) // 2, y_min
    elif direction == 'up':
      end_x, end_y = (x_min + x_max) // 2, y_max
    elif direction == 'right':
      end_x, end_y = x_min, (y_min + y_max) // 2
    elif direction == 'left':
      end_x, end_y = x_max, (y_min + y_max) // 2
    else:
      print('Invalid direction')
      return
    command = adb_utils.generate_swipe_command(
        int(start_x), int(start_y), int(end_x), int(end_y)
    )
    response = adb_utils.issue_generic_request(command, env)
    _check_adb_response(response, f'scroll({direction}): ({start_x}, {start_y}) -> ({end_x}, {end_y})', result)

  elif action.action_type == 'swipe':  # Inverse of scroll.
    if action.x_ >= 0 and action.y_ >= 0:
      start_x, start_y = action.x, action.y
      end_x, end_y = action.x_, action.y_
    else:
      screen_width, screen_height = screen_size
      mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
      direction = action.direction
      if direction == 'down':
        start_x, start_y = mid_x, 0
        end_x, end_y = mid_x, screen_height
      elif direction == 'up':
        start_x, start_y = mid_x, screen_height
        end_x, end_y = mid_x, 0
      elif direction == 'left':
        start_x, start_y = 0, mid_y
        end_x, end_y = screen_width, mid_y
      elif direction == 'right':
        start_x, start_y = screen_width, mid_y
        end_x, end_y = 0, mid_y
      else:
        print('Invalid direction')
        return
    command = adb_utils.generate_swipe_command(
        int(start_x), int(start_y), int(end_x), int(end_y), 500
    )
    response = adb_utils.issue_generic_request(command, env)
    _check_adb_response(response, f'swipe: ({start_x}, {start_y}) -> ({end_x}, {end_y})', result)

  elif action.action_type == 'open_app':
    app_name = action.app_name
    if app_name:
      response = adb_utils.launch_app(app_name, env)
      _check_adb_response(response, f'open_app({repr(app_name)})', result)
    else:
      raise ValueError('No app name provided')

  elif action.action_type == 'wait':
    time.sleep(1.0)

  elif action.action_type == 'launch_adb_activity':
    if action.activity_nickname == 'app_drawer':
      response = adb_utils.press_home_button(env)
      _check_adb_response(response, 'press_home_button (app_drawer)', result)
      time.sleep(1.0)
      start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
      end_x = start_x
      end_y = int(0.3 * screen_size[1])
      request = adb_utils.generate_swipe_command(start_x, start_y, end_x, end_y)
      response = adb_utils.issue_generic_request(request, env)
      _check_adb_response(response, f'swipe for app_drawer: ({start_x}, {start_y}) -> ({end_x}, {end_y})', result)
    elif action.activity_nickname == 'quick_settings':
      start_x, start_y = int(screen_size[0] / 2), 30
      end_x = start_x
      end_y = int(0.3 * screen_size[1])
      request = adb_utils.generate_swipe_command(
          start_x, start_y, end_x, end_y, duration_ms=10
      )
      response = adb_utils.issue_generic_request(request, env)
      _check_adb_response(response, f'swipe for quick_settings: ({start_x}, {start_y}) -> ({end_x}, {end_y})', result)
  elif action.action_type == 'change_orientation':
    response = adb_utils.change_orientation(action.orientation, env)
    _check_adb_response(response, f'change_orientation({action.orientation})', result)
  elif action.action_type == json_action.UNKNOWN:
    print('Unknown action type; no action will be executed. Try again...')
  else:
    print('Invalid action type')


def find_and_click_element(
    element_text: str,
    env: android_world_controller.AndroidWorldController,
    case_sensitive: bool = False,
):
  """Identifies element with element_text and clicks it.

  Args:
    element_text: Text of the UI element to click on.
    env: The Android env instance.
    case_sensitive: Whether to use case sensitivity when determining which UI
      element to tap.
  """
  # Wait a bit for UI to stabilize before searching for elements
  # This ensures the UI has finished rendering after any previous actions
  time.sleep(1.0)

  # Find text.
  action = _wait_and_find_click_element(element_text, env, case_sensitive)

  ui_elements = env.get_ui_elements()
  screen_size = (0, 0)  # Unused, but required.
  execute_adb_action(action, ui_elements, screen_size, env)


def _wait_and_find_click_element(
    target_text: str,
    env: android_world_controller.AndroidWorldController,
    case_sensitive: bool,
    dist_threshold: int = 1,  # Allow one character difference.
) -> json_action.JSONAction:
  """Wait for the screen to update until "element_text" appears."""
  ui_elements = env.get_ui_elements()
  element, distance = _find_target_element(
      ui_elements, target_text, case_sensitive
  )
  start = time.time()
  current = time.time()
  while current - start < 10:
    if distance <= dist_threshold:
      return json_action.JSONAction(action_type='click', index=element)
    # Sleep to give UI time to update and avoid busy-waiting
    time.sleep(0.5)
    ui_elements = env.get_ui_elements()
    element, distance = _find_target_element(
        ui_elements, target_text, case_sensitive
    )
    current = time.time()
  raise ValueError(f'Target text "{target_text}" not found.')


def _find_target_element(
    ui_elements: list[representation_utils.UIElement],
    target_text: str,
    case_sensitive: bool,
) -> tuple[int, int]:
  """Determine the UI element with the closest match to target_text, by looking at the `text` and `content_description` of each UI element."""
  best_match_index = -1
  lowest_distance = int(1e9)

  for i, element in enumerate(ui_elements):
    for attr in [element.text, element.content_description]:
      if attr is not None:
        if case_sensitive:
          distance = _levenshtein_distance(target_text, attr)
        else:
          distance = _levenshtein_distance(target_text.lower(), attr.lower())
        if distance < lowest_distance:
          lowest_distance = distance
          best_match_index = i

  return (best_match_index, lowest_distance)


def _levenshtein_distance(s1: str, s2: str) -> int:
  """Compute the Levenshtein distance between two strings."""
  if len(s1) < len(s2):
    s1, s2 = s2, s1

  if not s2:
    return len(s1)

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      substitutions = previous_row[j] + (c1 != c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row

  return previous_row[-1]