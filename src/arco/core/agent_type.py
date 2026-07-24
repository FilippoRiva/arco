import re


class AgentType(str):
    """
    An open-ended, string-based agent identifier.

    Behaves like a plain str (hashable, JSON-serializable, comparable),
    but new agent types can be defined anywhere just by instantiating
    AgentType("SomeName") — no need to touch this class.
    """
    _registry: dict[str, AgentType] = {}

    def __new__(cls, value: str) -> AgentType:
        # Records into the registry of AgentTypes
        if value in cls._registry:
            return cls._registry[value]
        instance = super().__new__(cls, value)
        cls._registry[value] = instance

        # Adds the property in capslock for the agent (AgentType("Retriever") adds AgentType.RETRIEVER)
        attr_name = re.sub(r"\W+", "_", value).strip("_").upper()
        if attr_name and not hasattr(cls, attr_name):
            setattr(cls, attr_name, instance)

        return instance

    @property
    def value(self) -> str:
        # keeps `.value` working anywhere the old Enum-style access is used
        return str(self)

    @classmethod
    def all(cls) -> list[AgentType]:
        """All agent types registered so far."""
        return list(cls._registry.values())
