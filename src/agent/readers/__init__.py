
"""Create model readers by name."""

from agent.readers.base import Answer, Reader, Request
from agent.readers.generated import GeneratedReader
from agent.readers.typed import TypedReader


READERS = {
    "typed": TypedReader,
    "generated": GeneratedReader,
    "generated_thinking": lambda model: GeneratedReader(model, thinking=True),
}


def make_reader(name: str, model: str) -> Reader:
    if name not in READERS:
        raise KeyError(f"Unknown reader {name!r}. Available: {list(READERS)}")
    return READERS[name](model)


__all__ = ["READERS", "make_reader", "Reader", "Request", "Answer"]
