from abc import ABC, abstractmethod
from typing import Dict, Any

class Trader(ABC):
    """Abstract base class for all trading signal generators."""

    @abstractmethod
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a trading signal dict given the current market state."""
        ...
