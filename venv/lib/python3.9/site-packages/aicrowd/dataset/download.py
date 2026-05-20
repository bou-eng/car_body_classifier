"""
The dataset download subcommand
"""

import logging
from typing import List

from aicrowd.contexts import ChallengeContext, ConfigContext
from aicrowd.dataset.exceptions import DatasetNotFoundException
from aicrowd.dataset.helpers import Downloader, get_file_indices, must_get_api_key
from aicrowd.dataset.list import list_dataset
from aicrowd.errors import INVALID_PARAMETER


def download_dataset(
    challenge: str,
    output_dir: str,
    jobs: int,
    dataset_files: List[str],
    config_ctx: ConfigContext = ConfigContext(),
    challenge_ctx: ChallengeContext = ChallengeContext(),
):
    """
    Downloads the specified datasets available for this challenge

    Args:
        challenge: one of

            - [`int`] challenge id
            - [`str`] challenge slug
            - [`str`] challenge url
        output_dir: where to download files, cwd by default
        jobs: how many files to download in parallel, (no. of cores)/2 by default
        dataset_files: which ones to download?, all by default
        config_ctx: CLI config
        challenge_ctx: Challenge config

    Example:
        To download all datasets for "example-challenge" into the current dir, 4 in parallel

        ```python
        from aicrowd import dataset
        dataset.download("example-challenge", "./", 4, [])
        ```
    """
    log = logging.getLogger()
    api_key = must_get_api_key(config_ctx)

    datasets = list_dataset(challenge, True, config_ctx, challenge_ctx)
    n_datasets = len(datasets)

    log.info("Got %d datasets", n_datasets)

    if len(dataset_files) == 0:
        dataset_files = list(range(len(datasets)))

    filtered_datasets = []

    dataset_file_indices = []
    for i in dataset_files:
        dataset_file_indices += get_file_indices(i, datasets)
    dataset_file_indices = list(set(dataset_file_indices))

    for file_index in dataset_file_indices:
        filtered_datasets.append(datasets[file_index])

    if len(filtered_datasets) == 0:
        raise DatasetNotFoundException(
            "No valid dataset found", "Run `aicrowd dataset list`", INVALID_PARAMETER
        )

    Downloader(jobs, api_key).download(
        map(lambda x: x.get("external_url"), filtered_datasets), output_dir
    )
