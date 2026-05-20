"""
Config subcommand
"""

import click

from aicrowd.contexts import ConfigContext, pass_config
from aicrowd.utils import AliasedGroup, CommandWithExamples
from aicrowd.utils.utils import exception_handler


@click.group(name="config", cls=AliasedGroup)
def config_command():
    """
    View and set config values
    """


@click.command(name="get", cls=CommandWithExamples)
@click.argument("key", type=str)
@pass_config
@exception_handler
def get_subcommand(config_ctx: ConfigContext, key: str):
    """
    View config values
    """
    from aicrowd.config import config_get

    print(config_get(key, config_ctx))


config_command.add_command(get_subcommand)
