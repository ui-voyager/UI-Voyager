"""Base LLM Client - Defines unified interface"""

import abc
import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Convert numpy array to JPEG bytes"""
    img = Image.fromarray(image)
    return image_to_jpeg_bytes(img)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.read()


def encode_image_base64(image: Union[np.ndarray, Image.Image]) -> str:
    """Encode image to base64 string"""
    if isinstance(image, np.ndarray):
        img_bytes = array_to_jpeg_bytes(image)
    else:
        img_bytes = image_to_jpeg_bytes(image)
    return base64.b64encode(img_bytes).decode('utf-8')


class BaseLLMClient(abc.ABC):
    """Base LLM Client - Supports text and multimodal requests"""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retry: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.retry_delay = retry_delay
    
    @abc.abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """Send chat request"""
        pass
    
    @abc.abstractmethod
    def multimodal_completion(
        self,
        text_prompt: str,
        images: List[np.ndarray],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """Send multimodal request (text + images)"""
        pass
    
    def predict(self, text_prompt: str, **kwargs) -> Tuple[str, Any]:
        """Convenience method for text prediction"""
        return self.chat_completion(
            messages=[{'role': 'user', 'content': text_prompt}],
            **kwargs
        )
    
    def predict_mm(
        self, 
        text_prompt: str, 
        images: List[np.ndarray],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """Convenience method for multimodal prediction"""
        return self.multimodal_completion(
            text_prompt=text_prompt,
            images=images,
            system_prompt=system_prompt,
            **kwargs
        )
