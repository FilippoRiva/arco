class AgentType(str):
    """Agent type identifier.

    Behaves exactly like a plain ``str`` — hashable, JSON-serializable,
    comparable, pickle-safe.  Use this type annotation wherever an
    agent type identifier is expected.

    Construct with ``AgentType("Retriever")``; the resulting value is
    a string that compares equal to ``"Retriever"``.
    """

    __slots__ = ()


__all__ = ["AgentType"]
