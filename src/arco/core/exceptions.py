from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AgentType


class AgentException(Exception):
    """This exception is risen if the Agent occurs in a fatal exception"""

    def __init__(
        self,
        message: str | None = None,
        missing_dependencies_from: AgentType | None = None,
        *args: object,
    ) -> None:
        if missing_dependencies_from:
            message = f"Missing a dependency from {missing_dependencies_from.value}"
        elif message is None:
            message = "Agent generated a fatal exception"
        super().__init__(message, args)


class EvaluatorException(AgentException):
    """Exception raised when evaluation fails."""


class ConfigException(Exception):
    """Raised when there's some fatal error in the usage of an ArcoConfig or AgentConfig"""


class StateException(Exception):
    """Raised when"""


__all__ = ["AgentException", "ConfigException", "EvaluatorException", "StateException"]
