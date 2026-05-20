"""
Exceptions for notebook subcommand
"""

from aicrowd.exceptions import CLIException


class NotebookException(CLIException):
    """
    Base exception for the notebook subcommand
    """


class NotebookAppImportException(NotebookException):
    """
    Exception for notebook app import errors
    """


class InvalidJupyterResponse(NotebookException):
    """
    Exception for bad responses from jupyter REST API
    """


class NotebookNotFound(NotebookException):
    """
    Exception for missing jupyter notebook
    """


class SubmissionFileNotFound(NotebookException):
    """
    Exception for missing submission zip file
    """


class AssetsDirectoryError(NotebookException):
    """
    Exception for assets directory validations
    """


class FeatureNotReady(NotebookException):
    """
    Exception for unshipped features
    """


class LocalEvaluationError(NotebookException):
    """
    Exception for local evaluation failure
    """


class NotebookFetchException(NotebookException):
    """
    Exception for failure while getting latest notebook content from frontend
    """
