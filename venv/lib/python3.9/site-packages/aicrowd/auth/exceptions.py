"""
Exceptions for login
"""

from aicrowd.exceptions import CLIException


class LoginException(CLIException):
    """
    Base exception for login command
    """


class CredentialException(LoginException):
    """
    credential verification exception
    """
