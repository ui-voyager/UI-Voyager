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

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
import numpy as np
from PIL import Image
import requests
from typing import List, Dict, Any, Optional


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """



  
  
class OpenAIWrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI client wrapper for Qwen3-VL via OpenAI-compatible API.

  Attributes:
    endpoint: The API endpoint URL.
    model_name: Model name to use (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct').
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    top_p: Top-p sampling parameter.
    max_tokens: Maximum tokens to generate.
    system_prompt: Optional system prompt to prepend to all requests.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      endpoint: str = 'http://localhost:8000/v1/chat/completions',
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.8,
      max_tokens: int = 2048,
      system_prompt: str | None = None,
  ):
    """Initializes the OpenAI wrapper.

    Args:
      model_name: Name of the model to use.
      endpoint: API endpoint URL.
      max_retry: Maximum number of retries on failure.
      temperature: Temperature for sampling.
      top_p: Top-p sampling parameter.
      max_tokens: Maximum tokens to generate.
      system_prompt: Optional system prompt for all requests.
    """
    try:
      from openai import OpenAI
    except ImportError:
      raise RuntimeError(
          'OpenAI package not installed. Install with: pip install openai'
      )

    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.model_name = model_name
    self.temperature = temperature
    self.top_p = top_p
    self.max_tokens = max_tokens
    self.system_prompt = system_prompt

    # Extract base URL from endpoint
    base_url = endpoint.rsplit('/chat/completions', 1)[0]
    self.client = OpenAI(base_url=base_url, api_key='EMPTY')

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    """Encode numpy array image to base64 string."""
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with text-only prompt."""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray], current_step: int = None
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with multimodal prompt (text + images).

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy arrays. The last image is the current
        observation, and all previous images are historical observations.
      current_step: Current step number K. Historical images will be labeled
        as steps K-N, K-N+1, ..., K-2, K-1 where N is the number of historical images.

    Returns:
      Tuple of (response_text, is_safe, raw_response).
    """
    if len(images) > 1:
      assert current_step is not None, (
          'current_step must be provided when there are historical images.'
      )
    assert len(images) > 0, 'At least one image must be provided.'
    # Build user content with text and images
    user_content = [{'type': 'text', 'text': text_prompt}]

    # Handle historical observations (all images except the last one)
    historical_images = images[:-1] if len(images) > 1 else []
    current_image = images[-1]

    # Add historical observations section if there are any
    if historical_images:
      user_content.append({
          'type': 'text',
          'text': f'Historical observations (You have observed the following {len(historical_images)} screen(s) on the current device):'
      })

      # Calculate step numbers for historical images
      # If current_step is K, historical images are K-N, K-N+1, ..., K-2, K-1
      n_historical = len(historical_images)
      for index, image in enumerate(historical_images):
        # Calculate the actual step number for this historical image
        step_number = current_step - n_historical + index

        # Add descriptive text for each historical observation
        user_content.append({
            'type': 'text',
            'text': f'Historical observation (Step {step_number}):'
        })

        # Add the historical image
        base64_image = self.encode_image(image)
        user_content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{base64_image}', 'detail': 'high'},
        })

    # Add current observation section
    if current_step is not None and len(historical_images) > 0:
      user_content.append({
          'type': 'text',
          'text': f'Current observation (Step {current_step}):'
      })

    base64_image = self.encode_image(current_image)
    user_content.append({
        'type': 'image_url',
        'image_url': {'url': f'data:image/jpeg;base64,{base64_image}', 'detail': 'high'},
    })

    # Build messages list
    messages = []
    if self.system_prompt:
      messages.append({
          'role': 'system',
          'content': [{'type': 'text', 'text': self.system_prompt}],
      })

    messages.append({
        'role': 'user',
        'content': user_content,
    })
    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        response_text = completion.choices[0].message.content
        return response_text, None, completion

      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
        if counter > 0:
          time.sleep(wait_seconds)
          wait_seconds *= 2

    return ERROR_CALLING_LLM, None, None


class GPT4VWrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4V wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    endpoint: API endpoint URL for the model service.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    top_p: Top-p (nucleus) sampling parameter.
    top_k: Top-k sampling parameter.
    seed: Random seed for reproducibility.
    repetition_penalty: Penalty for repeating tokens.
    presence_penalty: Penalty for token presence.
    max_tokens: Maximum number of tokens to generate.
    greedy: Whether to use greedy decoding (overrides temperature).
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      endpoint: str = 'http://0.0.0.0:22002/v1/chat/completions',
      max_retry: int = 3,
      temperature: float = 0.7,
      top_p: float = 0.8,
      top_k: int = 20,
      seed: int | None = 3407,
      repetition_penalty: float = 1.0,
      presence_penalty: float = 1.5,
      max_tokens: int = 32768,
      greedy: bool = False,
  ):
    # if 'OPENAI_API_KEY' not in os.environ:
    #   raise RuntimeError('OpenAI API key not set.')
    # self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.model_name = model_name
    self.endpoint = endpoint
    self.temperature = temperature
    self.top_p = top_p
    self.top_k = top_k
    self.seed = seed
    self.repetition_penalty = repetition_penalty
    self.presence_penalty = presence_penalty
    self.max_tokens = max_tokens
    self.greedy = greedy

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        # 'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        # 'model': self.model,
        'temperature': 0.0 if self.greedy else self.temperature,
        # 'top_p': self.top_p,
        # 'presence_penalty': self.presence_penalty,
        # 'max_tokens': self.max_tokens,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
    }

    # Add optional parameters directly to payload (not in extra_body)
    # if self.seed is not None:
    #   payload['seed'] = self.seed
    # if self.top_k is not None:
    #   payload['top_k'] = self.top_k
    # if self.repetition_penalty != 1.0:
    #   payload['repetition_penalty'] = self.repetition_penalty

    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}',
              # "max_pixels": 2_592_000   # 1080 x 2400
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
        )
        if response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None


class MGUIWrapper(LlmWrapper, MultimodalLlmWrapper):
  """MGUI wrapper for Qwen2.5-VL via OpenAI-compatible API.

  Attributes:
    endpoint: The API endpoint URL.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: Model name to use.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str | None = None,
      endpoint: str = 'http://29.181.192.11:8000/v1/chat/completions',
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    """Initializes the MGUI wrapper.

    Args:
      model_name: Name of the model to use. If None, will auto-detect from API.
      endpoint: API endpoint URL.
      max_retry: Maximum number of retries on failure.
      temperature: Temperature for sampling.
    """
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.endpoint = endpoint

    # Auto-detect model name if not provided
    if model_name is None:
      try:
        from openai import OpenAI
        # Extract base URL from endpoint
        base_url = endpoint.rsplit('/chat/completions', 1)[0]
        client = OpenAI(base_url=base_url, api_key='EMPTY')
        models = client.models.list()
        self.model = models.data[0].id
        print(f'Auto-detected model: {self.model}')
      except Exception as e:
        print(f'Failed to auto-detect model: {e}')
        self.model = 'Qwen/Qwen2.5-VL-7B-Instruct'  # fallback
        print(f'Using fallback model: {self.model}')
    else:
      self.model = model_name
      
class PhoneAgentWrapper(LlmWrapper, MultimodalLlmWrapper):
  """PhoneAgent wrapper for GLM via OpenAI-compatible API.

  Attributes:
    endpoint: The API endpoint URL.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: Model name to use.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str | None = None,
      endpoint: str = 'http://localhost:8000/v1/chat/completions',
      max_retry: int = 3,
      temperature: float = 0.1,
      frequency_penalty: float = 0.2,
      max_tokens: int = 3000,
  ):
    """Initializes the PhoneAgent wrapper.

    Args:
      model_name: Name of the model to use. If None, will auto-detect from API.
      endpoint: API endpoint URL.
      max_retry: Maximum number of retries on failure.
      temperature: Temperature for sampling.
      top_p: Top-p sampling parameter.
      frequency_penalty: Frequency penalty parameter.
      max_tokens: Maximum tokens to generate.
      skip_special_tokens: Whether to skip special tokens in output.
    """
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.frequency_penalty = frequency_penalty
    self.max_tokens = max_tokens
    self.endpoint = endpoint

    # Auto-detect model name if not provided
    if model_name is None:
      try:
        from openai import OpenAI
        # Extract base URL from endpoint
        base_url = endpoint.rsplit('/chat/completions', 1)[0]
        client = OpenAI(base_url=base_url, api_key='EMPTY')
        models = client.models.list()
        self.model = models.data[0].id
        print(f'Auto-detected model: {self.model}')
      except Exception as e:
        print(f'Failed to auto-detect model: {e}')
        self.model = 'autoglm-phone-9b'  # fallback
        print(f'Using fallback model: {self.model}')
    else:
      self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    """Encode numpy array image to base64 string."""
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with text-only prompt."""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with multimodal prompt (text + images).

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy arrays.

    Returns:
      Tuple of (response_text, is_safe, raw_response).
    """
    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'frequency_penalty': self.frequency_penalty,
        'max_tokens': self.max_tokens,
        'skip_special_tokens': False,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
    }

    # Add images to the content list
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        if response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        error_msg = response.json().get('error', {}).get('message', 'Unknown error')
        print(f'Error calling MGUI API with error message: {error_msg}')
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
        if counter > 0:
          time.sleep(wait_seconds)
          wait_seconds *= 2
    return ERROR_CALLING_LLM, None, None
