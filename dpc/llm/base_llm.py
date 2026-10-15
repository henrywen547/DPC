from abc import ABC, abstractmethod
from copy import deepcopy
import threading
from typing import List, Dict, Any, Optional, Union

class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0
        self.successful_api_calls = 0
        self.failed_api_calls = 0
        self.usage_reported_calls = 0
        self.prompt_trace = []
        self._usage_lock = threading.Lock()

    @abstractmethod
    def ask(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a list of messages (multi-turn) to the LLM.
        Returns the response text.
        Raises an exception if all retries fail.
        """
        pass

    def get_usage(self) -> Dict[str, int]:
        """Returns the total token usage so far."""
        with self._usage_lock:
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "api_calls": self.total_api_calls,
                "api_successful_calls": self.successful_api_calls,
                "api_failed_calls": self.failed_api_calls,
                "usage_reported_calls": self.usage_reported_calls,
            }

    def reset_usage(self):
        """Resets the token usage counters."""
        with self._usage_lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_api_calls = 0
            self.successful_api_calls = 0
            self.failed_api_calls = 0
            self.usage_reported_calls = 0
            self.prompt_trace = []

    def get_prompt_trace(self) -> List[Dict[str, Any]]:
        """Return the exact messages sent during the current pipeline run."""
        with self._usage_lock:
            return deepcopy(self.prompt_trace)

    def _record_prompt(self, messages: List[Dict[str, str]], attempt: int) -> int:
        with self._usage_lock:
            index = len(self.prompt_trace)
            self.prompt_trace.append({
                "model_id": self.model_name,
                "attempt_in_ask": attempt,
                "messages": deepcopy(messages),
                "request_status": "started",
            })
            return index

    def _finish_prompt(self, index: int, status: str) -> None:
        with self._usage_lock:
            self.prompt_trace[index]["request_status"] = status

    def _record_api_call(self, successful: bool) -> None:
        with self._usage_lock:
            self.total_api_calls += 1
            if successful:
                self.successful_api_calls += 1
            else:
                self.failed_api_calls += 1

    def _add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Accumulates token usage from a provider response."""
        with self._usage_lock:
            self.total_prompt_tokens += int(prompt_tokens or 0)
            self.total_completion_tokens += int(completion_tokens or 0)
            self.usage_reported_calls += 1
