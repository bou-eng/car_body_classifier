"""
The dataset list subcommand
"""

import logging
from typing import List, Union

from rich import box
from rich.console import Console
from rich.table import Column, Table

from aicrowd.constants import ChallengeConstants
from aicrowd.contexts import ChallengeContext, ConfigContext
from aicrowd.dataset.exceptions import ChallengeNotFoundException
from aicrowd.dataset.helpers import (
    get_datasets,
    humanize_size,
    must_get_api_key,
    parse_cli_challenge,
)
from aicrowd.errors import INVALID_PARAMETER


def list_dataset(
    challenge: str,
    return_list: bool = True,
    config_ctx: ConfigContext = ConfigContext(),
    challenge_ctx: ChallengeContext = ChallengeContext(),
) -> Union[None, List[dict]]:
    """
    Lists the datasets available for this challenge

    Args:
        challenge: one of

            - [`int`] challenge id
            - [`str`] challenge slug
            - [`str`] challenge url
        return_list: if true, will return the datasets instead of pretty printing
        config_ctx: CLI config
        challenge_ctx: Challenge config

    Returns:
        list of datasets if `return_list` was True; None otherwise
    """
    log = logging.getLogger()

    api_key = must_get_api_key(config_ctx)
    challenge_id = challenge_ctx.challenge.get(ChallengeConstants.CONFIG_ID_KEY)

    # couldn't get from config, try the --challenge option
    if challenge_id is None:
        challenge_id, _ = parse_cli_challenge(challenge, api_key)

    # still couldn't deduce challenge
    if challenge_id is None:
        log.error("Failed to parse challenge")
        raise ChallengeNotFoundException(
            "Challenge with the given details could not be found",
            exit_code=INVALID_PARAMETER,
        )

    datasets = get_datasets(challenge_id, api_key)
    log.info("Got %d datasets", len(datasets))

    if return_list:
        return datasets

    console = Console()
    table = Table(
        Column("#", max_width=2),
        "Title",
        "Description",
        Column(header="Size", justify="right"),
        title=f"Datasets for challenge #{challenge_id}",
        show_header=True,
        header_style="bold magenta",
        box=box.SQUARE,
    )

    empty_to_dash = lambda x: "-" if not x or len(x) == 0 else x

    key_mapper = [
        ("title", empty_to_dash),
        ("description", empty_to_dash),
        ("external_file_size", humanize_size),
    ]

    for i, ds in enumerate(datasets):
        vals = [str(i)]

        for key, mapper in key_mapper:
            vals.append(mapper(ds.get(key)))

        table.add_row(*vals)

    console.print(table, justify="left")
