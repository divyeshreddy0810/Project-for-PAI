from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class RegimeDetector(ABC):
    """Abstract base class for all regime detectors."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the detector on training data."""
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Any:
        """Return regime label(s) for the given data."""
        ...
