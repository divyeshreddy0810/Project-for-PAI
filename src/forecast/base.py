from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class Forecaster(ABC):
    """Abstract base class for all price forecasters."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the forecaster on training features and targets."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return price predictions for the given features."""
        ...
