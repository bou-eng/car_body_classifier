"""
Class describing a challenge
"""

import logging

from git import Repo
from git.exc import InvalidGitRepositoryError

from aicrowd.constants import ChallengeConstants
from aicrowd.errors import CONFIG_NOT_SET
from aicrowd.exceptions import CLIException


class Challenge:
    """
    CLI's interpretation of a challenge
    """

    def __init__(self):
        # TODO: implement __repr__/__str__ for this
        self.log: logging.Logger = logging.getLogger()
        self.info = None
        try:
            self.repo = Repo(".", search_parent_directories=True)
        except InvalidGitRepositoryError:
            self.repo = None

    def load(self):
        """
        Check git config for CLI stuff
        """
        try:
            with self.repo.config_reader() as git_conf:
                challenge_id = git_conf.get_value(
                    ChallengeConstants.CONFIG_SECTION_NAME,
                    ChallengeConstants.CONFIG_ID_KEY,
                )
                challenge_slug = git_conf.get_value(
                    ChallengeConstants.CONFIG_SECTION_NAME,
                    ChallengeConstants.CONFIG_SLUG_KEY,
                )

                self.info = {
                    ChallengeConstants.CONFIG_ID_KEY: challenge_id,
                    ChallengeConstants.CONFIG_SLUG_KEY: challenge_slug,
                }
                self.log.info("Read challenge config\n---\n%s\n---", self.info)
        except Exception as e:
            self.log.error("Error while reading the git config, %s", e)
            self.info = {}

    def get(self, key: str, ensure_exists: bool = False, help_msg: str = ""):
        """
        Gets a key from config

        Prints help_msg if ensure_exists is True and key not found

        Args:
            key: the config setting you are looking for
            ensure_exists: should this die if key doesn't exist?
            help_msg: how to fix this issue

        Returns:
            the value for the key
        """
        if self.info is None:
            self.load()

        value = self.info.get(key)

        if value is None and ensure_exists:
            self.log.warning("Queried challenge config for %s but not found", key)
            raise CLIException(
                f"Challenge config property {key} not set", help_msg, CONFIG_NOT_SET
            )

        return value
