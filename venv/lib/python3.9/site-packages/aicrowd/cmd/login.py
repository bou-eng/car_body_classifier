"""
For handling logins
"""

import logging

import click

from aicrowd.contexts import ConfigContext, pass_config
from aicrowd.utils.utils import exception_handler


@click.command(name="login")
@click.option(
    "--api-key", type=str, help="API Key from AIcrowd website", envvar="AICROWD_API_KEY"
)
@pass_config
@exception_handler
def login_command(config_context: ConfigContext, api_key: str):
    """
    Log in using AIcrowd API Key
    """
    from aicrowd.auth import aicrowd_login
    from aicrowd.auth.url_login import url_login

    log = logging.getLogger()

    gitlab_details = {}
    if api_key is None:
        log.info("API Key not provided in parameters, prompting")

        api_key, gitlab_details = url_login()

    aicrowd_login(api_key, gitlab_details, config_context)
