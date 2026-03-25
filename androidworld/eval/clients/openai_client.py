"""OpenAI-compatible API Client - Supports vLLM and other local deployments"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from eval.clients.base_client import BaseLLMClient, encode_image_base64


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible API Client"""
    
    def __init__(
        self,
        base_url: str = 'http://localhost:8000',
        model: str = 'Qwen3-VL-4B-Instruct',
        api_key: str = 'EMPTY',
        temperature: float = 0.7,
        max_tokens: int = 16384,
        top_p: float = 0.8,
        max_retry: int = 3,
        retry_delay: float = 2.0,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            retry_delay=retry_delay,
        )
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.top_p = top_p
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """Send chat request"""
        final_messages = []
        if system_prompt:
            final_messages.append({'role': 'system', 'content': system_prompt})
        final_messages.extend(messages)
        
        payload = {
            'model': kwargs.get('model', self.model),
            'messages': final_messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
        }
        
        return self._request_with_retry(payload)
    
    def multimodal_completion(
        self,
        text_prompt: str,
        images: List[np.ndarray],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """Send multimodal request    
        Supports using <image> placeholders in text_prompt to specify image insertion positions.
        """
        user_content = []
        
        if '<image>' in text_prompt:
            parts = text_prompt.split('<image>')
            image_idx = 0
            
            for i, part in enumerate(parts):
                if part:
                    user_content.append({'type': 'text', 'text': part})
                
                if i < len(parts) - 1 and image_idx < len(images):
                    base64_image = encode_image_base64(images[image_idx])
                    user_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}',
                            'detail': 'high',
                        },
                    })
                    image_idx += 1
            
            while image_idx < len(images):
                base64_image = encode_image_base64(images[image_idx])
                user_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}',
                        'detail': 'high',
                    },
                })
                image_idx += 1
        else:
            user_content.append({'type': 'text', 'text': text_prompt})
            
            for image in images:
                base64_image = encode_image_base64(image)
                user_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}',
                        'detail': 'high',
                    },
                })
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]})
        messages.append({'role': 'user', 'content': user_content})
        
        payload = {
            'model': kwargs.get('model', self.model),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
        }
        
        return self._request_with_retry(payload)
    
    def _request_with_retry(self, payload: Dict[str, Any]) -> Tuple[str, Any]:
        """Request with retry"""
        endpoint = f'{self.base_url}/v1/chat/completions'
        counter = self.max_retry
        delay = self.retry_delay
        
        while counter > 0:
            try:
                response = requests.post(endpoint, headers=self.headers, json=payload, timeout=120)
                
                if response.ok:
                    result = response.json()
                    if 'choices' in result:
                        text = result['choices'][0]['message']['content']
                        return text, result
                
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                print(f'API request failed: {error_msg}')
                
            except Exception as e:
                print(f'Request error: {e}')
            
            counter -= 1
            if counter > 0:
                print(f'Retrying in {delay}s...')
                time.sleep(delay)
                delay *= 2
        
        return 'Error calling LLM', None
