from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AgentType


class AgentException(Exception):
    """This exception is risen if the Agent occurs in a fatal exception"""

    def __init__(self,
                 message: str = "",
                 missing_answer_from_type: AgentType | None = None,
                 missing_dataframe_from_type: AgentType | None = None,
                 *args: object) -> None:
        if missing_dataframe_from_type and missing_answer_from_type:
            pass  # ignores the options if both are passed
        elif missing_answer_from_type:
            message = f"Missing a needed answer from {missing_answer_from_type.value}"
        elif missing_dataframe_from_type:
            message = f"Missing dataframe from {missing_dataframe_from_type.value}'s result"
        super().__init__(message, args)


class EvaluatorException(AgentException):
    """Exception raised when evaluation fails."""


class ConfigException(Exception):
    """Raised when there's some fatal error in the usage of an ArcoConfig or AgentConfig"""
