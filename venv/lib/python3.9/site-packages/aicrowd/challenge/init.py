"""
The challenge init subcommand
"""

import logging
from pathlib import Path

from git import Repo

from aicrowd.challenge.exceptions import ChallengeNotFoundException
from aicrowd.challenge.helpers import (
    create_challenge_dir,
    must_get_api_key,
    parse_cli_challenge,
    validate_challenge_dir,
)
from aicrowd.constants import ChallengeConstants
from aicrowd.contexts import ConfigContext
from aicrowd.errors import INVALID_PARAMETER


def init_challenge(
    challenge: str,
    base_dir: str,
    mkdir: bool,
    config_ctx: ConfigContext = ConfigContext(),
):
    """
    Setups the basic challenge files

    Args:
        challenge: one of

            - [`int`] challenge id
            - [`str`] challenge slug
            - [`str`] challenge url
        base_dir: directory for setting up challenge
        mkdir: should the base directory be created if it doesn't exist?
        config_ctx: CLI Config

    Example:
        To create a challenge directory named `example` for `example-challenge`

        ```python
        from aicrowd import challenge
        challenge.init("example-challenge", "example", True)
        ```
    """
    log = logging.getLogger()

    api_key = must_get_api_key(config_ctx)

    challenge_id, challenge_slug = parse_cli_challenge(challenge, api_key)

    if challenge_id is None or challenge_slug is None:
        log.error("Failed to parse challenge")
        raise ChallengeNotFoundException(
            "Challenge with the given details could not be found",
            exit_code=INVALID_PARAMETER,
        )

    # create the challenge directory
    if base_dir is None:
        challenge_dir = Path.cwd()
    else:
        challenge_dir = base_dir

    if mkdir:
        challenge_dir = create_challenge_dir(challenge_dir, challenge_slug)

    if not validate_challenge_dir(challenge_dir):
        raise ChallengeNotFoundException(
            "Challenge dir should be empty and not reside within an existing git repository.",
            exit_code=INVALID_PARAMETER,
        )

    repo = Repo.init(challenge_dir)

    with repo.config_writer() as git_conf:
        git_conf.add_value(
            ChallengeConstants.CONFIG_SECTION_NAME,
            ChallengeConstants.CONFIG_ID_KEY,
            challenge_id,
        )
        git_conf.add_value(
            ChallengeConstants.CONFIG_SECTION_NAME,
            ChallengeConstants.CONFIG_SLUG_KEY,
            challenge_slug,
        )
