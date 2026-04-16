from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModule(ABC):
    """Abstract base class for all processing modules"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled: bool = config.get("enabled", True)

    @abstractmethod
    def process(self, image_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single image, update and return the record dict.
        record already contains basic fields (filename, file_path, etc.).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for logging"""
        raise NotImplementedError
