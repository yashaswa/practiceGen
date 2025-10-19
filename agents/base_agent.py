from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentContext:
    doc_id: str
    filename: str


class BaseAgent:
    def __init__(self, context: Optional[AgentContext] = None):
        self.context = context

    def set_context(self, context: AgentContext):
        self.context = context