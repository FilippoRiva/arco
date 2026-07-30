import re
from typing import ClassVar, Self


class AgentType(str):
    """An open-ended, string-based agent identifier.

    Behaves like a plain ``str`` (hashable, JSON-serializable, comparable),
    but new agent types can be defined anywhere just by calling
    :meth:`register` — no need to modify this class.

    Auto-registered by :class:`Agent.__init_subclass__` when a concrete
    subclass is defined.
    """

    _registry: ClassVar[dict[str, AgentType]] = {}

    @classmethod
    def register(cls, value: str):
        """Register a new agent type.

        Creates a singleton instance and adds an uppercase class attribute
        (e.g. ``register("Retriever")`` adds ``AgentType.RETRIEVER``).

        :param value: The agent type name (e.g. ``"Retriever"``).
        """
        if value in cls._registry:
            return
        instance = super().__new__(cls, value)
        cls._registry[value] = instance

        attr_name = re.sub(r"\W+", "_", value).strip("_").upper()
        if attr_name and not hasattr(cls, attr_name):
            setattr(cls, attr_name, instance)

    def __new__(cls, value: str) -> Self:
        return cls._registry[value]

    @property
    def value(self) -> str:
        """Return the raw string value (for Enum-style access)."""
        return str(self)

    @classmethod
    def all(cls) -> list[AgentType]:
        """Return all registered agent types."""
        return list(cls._registry.values())


__all__ = ["AgentType"]
