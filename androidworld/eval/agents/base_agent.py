"""Base Agent - Defines unified agent interface"""

import abc
from typing import Any, Dict, Optional

from android_world.agents import base_agent
from android_world.env import interface

AgentStepResult = base_agent.AgentInteractionResult


class BaseEvalAgent(base_agent.EnvironmentInteractingAgent):
    """Base Evaluation Agent - Inherits from EnvironmentInteractingAgent for compatibility"""
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = 'EvalAgent',
        transition_pause: float = 1.0,
    ):
        super().__init__(env, name, transition_pause)
        self._history = []
    
    def reset(self, go_home: bool = False) -> None:
        """Reset agent state"""
        super().reset(go_home)
        self._history = []
    
    @abc.abstractmethod
    def step(self, goal: str, task_name: Optional[str] = None) -> AgentStepResult:
        """Execute one step"""
        pass
    
    def finalize_task(self, success: bool) -> None:
        """Cleanup when task ends"""
        pass
