"""
all api calls related to auth
"""

import logging
from http import HTTPStatus
from time import sleep
from typing import Union

from aicrowd.constants import GITLAB_HOST
from aicrowd_api.request import AIcrowdAPI, GitlabAPI, RailsAPI

log = logging.getLogger()


def verify_api_key(api_key: str) -> bool:
    """
    Verifies if the API Key is valid or not

    Args:
        api_key: AIcrowd API Key

    Returns:
        True if API Key valid, False otherwise
    """
    r = RailsAPI(api_key).get("/api_user")

    if not r.ok:
        log.error(
            "Error in verifying API Key.\nReason: %s, Message: %s", r.reason, r.text
        )

    return r.ok


def verify_gitlab_oauth_token(oauth_token: str) -> bool:
    r = GitlabAPI(base_url=f"https://{GITLAB_HOST}").get(
        "/oauth/token/info", {"access_token": oauth_token}
    )

    if not r.ok:
        log.error(
            "Error in verifying gitlab access token.\nReason: %s, Message: %s",
            r.reason,
            r.text,
        )

    return r.ok


def initiate_api_url_auth() -> Union[None, dict]:
    """
    Gets the login url and OTP

    Returns:
        login url and otp
    """
    r = AIcrowdAPI().get("/auth/initiate")

    if not r.ok:
        log.error(
            "Error in getting login details.\nReason: %s, Message: %s", r.reason, r.text
        )

        return None

    return r.json()


def get_user_details(
    random_key: str, otp: str, max_retries: int, poll_interval: int
) -> Union[dict, None]:
    """
    Tries to fetch login information
    Blocking call, waits till details are available or max_retries are over
    """
    for _ in range(max_retries):
        r = AIcrowdAPI().get(
            f"/auth/{random_key}/details", headers={"Authorization": otp}
        )

        if r.status_code != HTTPStatus.OK:
            sleep(poll_interval)
            log.info("Status code %d.\nResponse: %s", r.status_code, r.text)
        else:
            return r.json()["details"]

    return None
