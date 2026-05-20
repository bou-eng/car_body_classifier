"""
Exceptions for config subcommand command
"""

from aicrowd.exceptions import CLIException


class ConfigKeyNotFound(CLIException):
    """
    Config key not found
    """
