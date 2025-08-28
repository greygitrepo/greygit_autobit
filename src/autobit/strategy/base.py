from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def on_tick(self, tick: dict) -> None:
        pass

    @abstractmethod
    def on_bar(self, bar: dict) -> None:
        pass

