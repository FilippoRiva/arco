from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AgentType


class AgentException(Exception):
    """Raised when an agent encounters a fatal error."""

    def __init__(
        self,
        message: str | None = None,
        missing_dependencies_from: AgentType | None = None,
        *args: object,
    ) -> None:
        if missing_dependencies_from:
            message = f"Missing a dependency from {missing_dependencies_from}"
        elif message is None:
            message = "Agent generated a fatal exception"
        super().__init__(message, args)


class EvaluatorException(AgentException):
    """Raised when evaluation fails."""


class ConfigException(Exception):
    """Raised when there is a fatal error in the usage of a Config or AgentConfig."""


class StateException(Exception):
    """Raised when a State operation fails."""


__all__ = ["AgentException", "ConfigException", "EvaluatorException", "StateException"]
