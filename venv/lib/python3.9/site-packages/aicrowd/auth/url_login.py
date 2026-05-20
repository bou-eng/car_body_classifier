"""
AIcrowd login via url
"""

import logging
from json.decoder import JSONDecodeError
from typing import Dict, Tuple

import click

from aicrowd.auth.exceptions import LoginException
from aicrowd.constants import LoginConstants
from aicrowd_api.auth import get_user_details, initiate_api_url_auth


def url_login(poll_interval=5) -> Tuple[str, Dict]:
    """
    Fetches the login URL and OTP
    Polls till user authenticates
    """
    log = logging.getLogger()

    login_details = initiate_api_url_auth()

    if login_details is None:
        raise LoginException(
            "Couldn't contact AIcrowd servers", "Please contact AIcrowd administrators"
        )

    try:
        url, otp = login_details["login_url"], login_details["otp"]
    except KeyError as e:
        log.error("Couldn't extract expected keys login_url and otp from api response")
        raise LoginException(
            "Invalid response from AIcrowd servers",
            "Please contact AIcrowd administrators",
        ) from e

    click.echo(
        "Please login here: " + click.style(url, fg="blue", underline=True, bold=True)
    )
    click.launch(url)

    random_key = url.split("/")[-1]

    try:
        user_details = get_user_details(
            random_key,
            otp,
            max_retries=LoginConstants.URL_LOGIN_REQUEST_LIFETIME // poll_interval,
            poll_interval=poll_interval,
        )

        if user_details is None:
            raise LoginException(
                "Couldn't login. Max retries exceeded", "Please try logging in again"
            )

        gitlab_details = {
            "oauth_token": user_details.get("gitlab_oauth_token", ""),
            "userid": user_details.get("gitlab_userid", ""),
            "username": user_details.get("gitlab_username", ""),
        }

        return user_details["api_key"], gitlab_details
    except (JSONDecodeError, KeyError) as e:
        log.error("Couldn't extract API Key from response")
        raise LoginException(
            "Invalid response from AIcrowd servers",
            "Please contact AIcrowd administrators",
        ) from e
