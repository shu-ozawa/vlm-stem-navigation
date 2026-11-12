"""Connect the reader and policy, and maintain the history of an episode."""

from agent import policy
from agent.prompts.build import log_entry
from agent.readers import Reader


class Agent:
    def __init__(self, reader: Reader):
        self.reader = reader
        self.target = ""
        self.history = []

    @property
    def name(self) -> str:
        return f"{self.reader.name}/{self.reader.model}"

    def reset(self, target: str) -> None:
        """Start a new episode."""
        self.target = target
        self.history = []

    def act(self, obs: bytes, state: dict, inspect: bool = False) -> dict:
        """Decide the next action using the current image and operator state."""
        return policy.decide(
            obs, self.target, state, self.history, self.reader, inspect
        )

    def record(self, state: dict, decision: dict, blocked: bool) -> None:
        """Append the executed action and its result to the history."""
        self.history.append(
            log_entry(state, decision["choice"], blocked, decision)
        )