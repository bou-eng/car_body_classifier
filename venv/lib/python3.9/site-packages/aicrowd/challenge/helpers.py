"""
Challenge related tasks
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Tuple, Union

import click
import git

from aicrowd.auth.exceptions import CredentialException
from aicrowd.auth.helpers import verify_api_key
from aicrowd.challenge.exceptions import ChallengeNotFoundException
from aicrowd.constants import RAILS_HOST, LoginConstants
from aicrowd.contexts import ConfigContext
from aicrowd_api.challenge import get_challenge_details


def must_get_api_key(config_ctx: ConfigContext) -> str:
    """
    Checks whether API Key is defined in the config, prints a help message otherwise
    Also verifies that the API Key is valid

    Args:
        config_ctx: CLI Config

    Returns:
        AIcrowd API Key
    """
    api_key = config_ctx.config.get(
        LoginConstants.CONFIG_KEY,
        ensure_exists=True,
        help_msg=" ".join(
            [
                click.style("Please run", fg="red"),
                click.style("aicrowd login", fg="yellow"),
                click.style("to login.", fg="red"),
            ]
        ),
    )

    if not verify_api_key(api_key):
        raise CredentialException(
            "Your API Key is invalid", "Please run `aicrowd login` again"
        )

    return api_key


def parse_cli_challenge(challenge: Union[int, str], api_key: str) -> Tuple[int, str]:
    """
    Parses the cli input to extract challenge

    Args:
        challenge: one of

            - [`int`] challenge id
            - [`str`] challenge slug
            - [`str`] challenge url
        api_key: aicrowd api key

    Returns:
        challenge_id, challenge_slug
    """
    log = logging.getLogger()

    if challenge is None:
        return None, None

    if challenge.isnumeric():
        # id was provided
        challenge_id = int(challenge)
        challenge_slug = get_slug_from_id(challenge_id, api_key)
    else:
        if challenge.startswith("http"):
            challenge_slug = get_slug_from_url(challenge)

            if challenge_slug is None:
                log.warning(
                    "Couldn't parse slug from given URL, assuming slug was provided"
                )
                # just in case the challenge slug is like "http-challenge"
                challenge_slug = challenge
        else:
            # assume that slug was provided
            challenge_slug = challenge

        challenge_id = get_id_from_slug(challenge_slug, api_key)

    return challenge_id, challenge_slug


def get_challenge_with_params(key: str, value: str, api_key: str) -> dict:
    """
    Queries AIcrowd API for challenge where key=value

    Args:
        key: database column for the challenge
        value: value of that key for the challenge
        api_key: aicrowd api key

    Returns:
        challenge information
    """
    log = logging.getLogger()

    r = get_challenge_details(api_key, {key: value})

    if not r.ok:
        log.error("Request to API failed\nReason: %s\nMessage: %s", r.reason, r.text)
        raise ChallengeNotFoundException(
            f"Challenge with the given details could not be found. {r.reason}"
        )

    try:
        return r.json()[0]
    except (json.JSONDecodeError, IndexError) as e:
        log.error("Parsing response failed\n%s", e)
        raise ChallengeNotFoundException(
            "Error in decoding the response from AIcrowd servers",
            "Please recheck the challenge name",
        ) from e


def get_id_from_slug(slug: str, api_key: str) -> int:
    """
    Gets the challenge id from slug (if valid)

    Args:
        slug: challenge slug
        api_key: aicrowd api key

    Returns:
        challenge id
    """
    return get_challenge_with_params("slug", slug, api_key).get("id")


def get_slug_from_id(challenge_id: int, api_key: str) -> str:
    """
    Gets the challenge slug from id (if valid)

    Args:
        challenge_id: challenge id
        api_key: aicrowd api key

    Returns:
        challenge slug
    """
    return get_challenge_with_params("id", challenge_id, api_key).get("slug")


def get_slug_from_url(url: str) -> Union[str, None]:
    """
    Extracts slug from challenge url

    Args:
        url: challenge url

    Returns:
        challenge slug
    """
    log = logging.getLogger()

    slug_re = re.compile(f"https://{RAILS_HOST}/challenges/([^/]*)")
    slug_match = slug_re.match(url)

    if slug_match is None:
        log.error("Wrong url given? No match found")
        return None

    return slug_match.groups()[0]


def create_challenge_dir(challenge_base_dir: Path, challenge_slug: str) -> str:
    """
    Creates a new challenge directory with the help of slug

    Args:
        challenge_base_dir: base directory for creating the challenge
        challenge_slug:

    Returns:
        path to the created directory
    """
    log = logging.getLogger()
    dir_path = os.path.join(challenge_base_dir, challenge_slug)

    if os.path.exists(dir_path):
        suffix = 0
        log.warning(
            "Challenge directory %s already exists. A new one will be created", dir_path
        )

        while os.path.exists(f"{dir_path}-{suffix}"):
            suffix += 1

        dir_path += f"-{suffix}"

    Path(dir_path).mkdir(parents=True)
    return dir_path


def validate_challenge_dir(challenge_dir: str) -> bool:
    """
    Checks whether the challenge_dir is an empty directory outside git repo

    Args:
        challenge_dir: directory to be used for setting up challenge

    Returns:
        is challenge_dir valid?
    """
    log = logging.getLogger()

    if os.listdir(challenge_dir):
        log.error("Challenge dir is not empty")
        return False

    try:
        git.Repo(search_parent_directories=True)
        # if its a git repo, it's not valid
        return False
    except git.InvalidGitRepositoryError:
        return True
