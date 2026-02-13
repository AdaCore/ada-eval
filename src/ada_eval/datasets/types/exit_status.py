"""Exit status types for generation runs."""

from enum import StrEnum


class CommonExitCodes:
    TIMEOUT = 124
    CLI_NOT_FOUND = 127


class ExitStatus(StrEnum):
    """Exit status for generation runs."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    CLI_NOT_FOUND = "cli_not_found"
    PROCESS_ERROR = "process_error"
    SDK_ERROR = "sdk_error"
    UNKNOWN = "unknown"

    @classmethod
    def from_exit_code(cls, exit_code: int) -> "ExitStatus":
        """
        Map a shell exit code to an ExitStatus.

        Args:
            exit_code: The shell exit code.

        Returns:
            The corresponding ExitStatus.

        """
        match exit_code:
            case 0:
                return cls.SUCCESS
            case CommonExitCodes.TIMEOUT:
                return cls.TIMEOUT
            case CommonExitCodes.CLI_NOT_FOUND:
                return cls.CLI_NOT_FOUND
            case _:
                return cls.PROCESS_ERROR
