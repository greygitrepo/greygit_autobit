from __future__ import annotations

from abc import ABC, abstractmethod


class RiskManager(ABC):
    @abstractmethod
    def validate_order(self, order: dict) -> bool:
        pass

