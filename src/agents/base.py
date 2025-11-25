from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models import ComplianceState


class Agent(ABC):
    name: str = "BaseAgent"

    def __call__(self, state: ComplianceState) -> ComplianceState:
        setattr(self, "_log_written", False)
        result = self.run(state)
        if not getattr(self, "_log_written", False):
            result.log.append({"agent": self.name, "status": "completed"})
        return result

    @abstractmethod
    def run(self, state: ComplianceState) -> ComplianceState:
        raise NotImplementedError

