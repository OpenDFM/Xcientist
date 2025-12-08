from pydantic import BaseModel
from typing import Any

class BaseDictModel(BaseModel):
    """
    Base class for all agent output schemas to provide dictionary-like access.
    """
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

