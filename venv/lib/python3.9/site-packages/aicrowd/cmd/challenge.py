"""
Challenge subcommand
"""

import sys

import click

from aicrowd.contexts import ConfigContext, pass_config
from aicrowd.utils import AliasedGroup
from aicrowd.utils.utils import exception_handler


@click.group(name="challenge", cls=AliasedGroup)
def challenge_command():
    """
    Setup a challenge
    """


@click.command(name="init")
@click.argument("challenge", type=str)
@click.option(
    "-d",
    "--base-dir",
    type=click.Path(exists=True),
    help="Base directory for storing the challenge",
)
@click.option(
    "--mkdir",
    is_flag=True,
    help="Create a new directory for challenge inside current directory",
)
@pass_config
@exception_handler
def init_subcommand(
    config_ctx: ConfigContext, challenge: str, base_dir: str, mkdir: bool
):
    """
    Setups basic challenge files
    """
    from aicrowd.challenge import init_challenge

    init_challenge(challenge, base_dir, mkdir, config_ctx)


challenge_command.add_command(init_subcommand)
