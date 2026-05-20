"""
This class describes the config for the CLI
"""

import logging
import os
from pathlib import Path
from typing import Dict, Union

import click
import toml

from aicrowd.errors import CONFIG_NOT_SET, INVALID_FILE
from aicrowd.exceptions import CLIException


class CLIConfig:
    """
    Stores all the configuration items
    """

    def __init__(self):
        self.log: logging.Logger = logging.getLogger()
        self.config_path: str = None
        self._settings: Dict = None

    def load(self, config_path: str):
        """
        Loads the config file

        Args:
            config_path: path to config file
        """
        if config_path is None:
            self.config_path = os.path.join(
                click.get_app_dir("aicrowd-cli"), "config.toml"
            )
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config_path).touch()
        elif os.path.isfile(config_path):
            self.config_path = config_path
        else:
            self.log.error("Specified config file, '%s' doesn't exist", config_path)
            raise CLIException(
                f"Config file '{config_path}' doesn't exist", exit_code=INVALID_FILE
            )

        self.log.info("Loading config from %s", self.config_path)

        try:
            with open(self.config_path, "r") as config:
                self._settings = toml.load(config)
                self.log.info("Config loaded")
        except toml.TomlDecodeError:
            self.log.error(
                "Malformed config file: %s. Will be overwritten", self.config_path
            )
            self._settings = {}

    def write(self):
        """
        Save config to file
        """
        # make sure parent dir exists
        basedir = Path(self.config_path).parent.absolute()
        Path(basedir).mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w+") as config_file:
            toml.dump(self._settings, config_file)
            self.log.info(
                "Successfully saved config to %s\n---\n%s\n---",
                self.config_path,
                self._settings,
            )

    def get(
        self, key: str, ensure_exists: bool = False, help_msg: str = ""
    ) -> Union[str, Dict]:
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
        if self._settings is None:
            self.load(self.config_path)

        value = self._settings.get(key)

        if value is None and ensure_exists:
            self.log.warning("Queried config for %s but not found", key)
            raise CLIException(
                f"Config property {key} not set", help_msg, CONFIG_NOT_SET
            )

        return value

    def set(self, key: str, value: Union[str, Dict]):
        """
        Sets a key to value in config

        Args:
            key: config key
            value: config value
        """
        if self._settings is None:
            self.load(self.config_path)

        self._settings[key] = value
