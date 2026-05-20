"""
The submission create subcommand
"""

import logging
from typing import Optional

from aicrowd.constants import ChallengeConstants
from aicrowd.contexts import ChallengeContext, ConfigContext
from aicrowd.errors import INVALID_CHALLENGE_REPO, INVALID_PARAMETER
from aicrowd.submission.exceptions import (
    ChallengeNotFoundException,
    InvalidChallengeDirException,
)
from aicrowd.submission.helpers import (
    must_get_api_key,
    parse_cli_challenge,
    submit_file,
    submit_git,
    verify_file_submission,
    verify_git_submission,
)


def create_submission(
    challenge: str,
    file_path: str,
    description: str,
    print_links: bool = False,
    submission_tag: Optional[str] = None,
    config_ctx: ConfigContext = ConfigContext(),
    challenge_ctx: ChallengeContext = ChallengeContext(),
    no_verify: bool = False,
    **extra_kwargs,
):
    """
    Creates a submission on AIcrowd

    Considers both cases:

     - git submission
     - artifact (file based) submission

    if `-f/--file` is specified, they are submitting a file.
    So, default to artifact based submission

    Otherwise, default to gitlab based submission
      (should ignore challenge parameter in that case, pick up stuff from gitconfig)

    Args:
        challenge: one of

            - [`int`] challenge id
            - [`str`] challenge slug
            - [`str`] challenge url
        file_path: file to submit
        description: description for the submission
        print_links: print helpful links related to the submission
        submission_tag: [Git submissions] the tag to push
        jupyter: Bundles jupyter notebook submission
        config_ctx: CLI config
        challenge_ctx: Challenge config
    """
    log = logging.getLogger()

    api_key = must_get_api_key(config_ctx)
    challenge_id = challenge_ctx.challenge.get(ChallengeConstants.CONFIG_ID_KEY)

    # not in a valid challenge git directory and file not given
    if challenge_id is None and file_path is None:
        log.error("Not in a valid challenge git directory and file not given")
        raise InvalidChallengeDirException(
            "Please run this command from the challenge directory for git based submissions "
            + "or specify the file using -f/--file for artifact based submissions",
            exit_code=INVALID_PARAMETER,
        )

    # couldn't get from config, try the --challenge option
    if challenge_id is None:
        challenge_id, challenge_slug = parse_cli_challenge(challenge, api_key)

    # still couldn't deduce challenge
    if challenge_id is None:
        raise ChallengeNotFoundException(
            "Challenge with the given details could not be found",
            "Please recheck the challenge name",
            exit_code=INVALID_PARAMETER,
        )

    if file_path is None:
        if not no_verify:
            verify_git_submission(api_key, challenge_id)
        if not challenge_ctx.challenge.repo:
            raise InvalidChallengeDirException(
                "Couldn't find a git repo at the challenge location",
                exit_code=INVALID_CHALLENGE_REPO,
            )
        return submit_git(
            challenge_ctx.challenge.repo, description, submission_tag, print_links
        )

    if not no_verify:
        verify_file_submission(api_key, challenge_id)

    # challenge_slug is not unbound
    # we'll reach here only if initial challenge_id isn't set and we call parse_cli_challenge
    return submit_file(
        challenge_slug, file_path, description, api_key, print_links, extra_kwargs
    )
