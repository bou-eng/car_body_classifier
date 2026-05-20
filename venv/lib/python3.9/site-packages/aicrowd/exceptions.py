"""
Exceptions that the CLI raises

All other exceptions will inherit from `CLIException` so just catching that should be enough
"""

import re

import click

from aicrowd import errors


class CLIException(Exception):
    """
    Base class for all exceptions raised by CLI
    """

    def __init__(
        self, message: str, fix: str = None, exit_code: int = errors.UNKNOWN_ERROR
    ):
        """
        Args:
            message: this will be printed to the console
            fix: (if defined) what steps can be taken to fix the error
            exit_code: what the exit code should be if this exception was raised
        """
        super().__init__(message)

        self.message = message
        self.fix = fix
        self.exit_code = exit_code

    def __str__(self):
        exception_name = type(self).__name__
        exception_suffix = "Exception"

        if exception_name.endswith(exception_suffix):
            exception_name = exception_name[: -len(exception_suffix)]

        # https://stackoverflow.com/a/9283563
        # Convert PascalCase to Human Readable form smartly
        exception_name = re.sub(
            r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", exception_name
        )

        if self.fix:
            fix = click.style(f"\n{self.fix}", fg="yellow")
        else:
            fix = ""

        return click.style(f"{exception_name} Error: {self.message}", fg="red") + fix
