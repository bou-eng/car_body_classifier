"""
Dataset related tasks
"""

import fnmatch
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http import HTTPStatus
from typing import Dict, Iterable, List
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import click

# pylint: disable=unused-import
from aicrowd.challenge.helpers import must_get_api_key, parse_cli_challenge

# pylint: enable=unused-import
from aicrowd.constants import DATASETS_HOST
from aicrowd.utils import TqdmProgressBar
from aicrowd.utils.jupyter import is_google_colab_env
from aicrowd_api.challenge import get_challenge_datasets


def get_datasets(challenge_id: int, api_key: str) -> List[dict]:
    """
    Queries AIcrowd API for datasets of this challenge

    Args:
        challenge_id: challenge id
        api_key: AIcrowd API Key

    Returns:
        Datasets for a particular challenge
    """
    log = logging.getLogger()

    r = get_challenge_datasets(api_key, challenge_id)

    if not r.ok:
        log.error("Request to API failed\nReason: %s\nMessage: %s", r.reason, r.text)
        return [{}]

    try:
        return r.json()
    except Exception as e:
        log.error("Parsing response failed\n---\n%s\n---", e)
        return [{}]


def get_file_indices(
    picked_file: str, dataset_files: List[Dict[str, str]]
) -> List[int]:
    """
    Returns the index of the picked file in the dataset if the provided input is not an integer.

    Args:
        picked_file: glob pattern for files to be downloaded
        dataset_files:
    """
    try:
        idx = int(picked_file)
        return [idx]
    except ValueError:
        matched_indices = []
        for idx, dataset_file in enumerate(dataset_files):
            if fnmatch.fnmatch(dataset_file.get("title", ""), picked_file):
                matched_indices.append(idx)
    return matched_indices


def humanize_size(size: int) -> str:
    """
    Returns the file size (inp=bytes) in a human readable format

    Args:
        size: size in bytes

    Returns:
        size in human readable format
    """
    try:
        size = float(size)
    except ValueError:
        return size

    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(size) < 1000:
            return f"{size:.2f} {unit}"

        size /= 1000

    return f"{size:.2f} YB"


class Downloader:
    """
    A rudimentary URL downloader (like wget or curl) to demonstrate Rich progress bars.

    Adapted from [here](https://github.com/willmcgugan/rich/blob/master/examples/downloader.py)
    """

    def __init__(
        self, njobs: int = None, api_key: str = None, chunk_size: int = 1024 * 1024
    ):
        """
        Args:
            njobs: number of parallel downloads
            api_key: AIcrowd API Key
            chunk_size: no. of bytes to process in each iteration
        """
        self.progress_bar = TqdmProgressBar()

        if njobs is None:
            njobs = max(os.cpu_count() // 2, 1)

        self.njobs = njobs
        self.api_key = api_key
        # need this to indicate when the thread should die
        self.__active = False
        self.chunk_size = self.get_chunk_size(chunk_size)

        self.log = logging.getLogger()

    @staticmethod
    def get_chunk_size(chunk_size: int) -> int:
        """
        Set chunk size of 100 MB if running on Google colab

        Args:
            chunk_size: default size to return if not on colab
        """
        if is_google_colab_env():
            return 32 * 1024 * 1024

        return chunk_size

    def copy_url(self, filename: str, url: str, path: str) -> None:
        """
        Copy data from a url to a local file

        Args:
            url: url for the file to be downloaded
            path: path where the file will be saved
        """
        req = Request(url)
        is_dataset_on_aicrowd = urlparse(url).netloc == DATASETS_HOST

        # datasets.aicrowd.com checks whether participant is registered
        # for the challenge using the api key provided
        if is_dataset_on_aicrowd:
            # IMPORTANT: must use unredirected header
            # 1. security, random places won't end up with aicrowd api key
            # 2. this sometimes gets redirected to aws s3 which is supposed to have its
            #       own Authorization token and keeping this causes issues
            req.add_unredirected_header("Authorization", f"Token {self.api_key}")

        try:
            response = urlopen(req)
        except HTTPError as e:
            self.log.error("Error in downloading dataset %s.\n%s", url, e)
            click.echo(click.style(f"Error in downloading dataset {url}", fg="red"))

            # only possible when:
            # 1. participant isn't registered for that challenge
            # 2. it is a private file (configuration error somewhere)
            if is_dataset_on_aicrowd and e.status == HTTPStatus.UNAUTHORIZED:
                # for type #1
                error_msg = "Please make sure that you are registered for the challenge"

                response = e.fp.read().decode()
                registration_link = re.findall("href=([^>]*)", response)
                if len(registration_link) > 0:
                    error_msg += f": {registration_link[0]}"

                # for type #2
                error_msg += (
                    "\nIf the issue still persists, please contact challenge organizers"
                )

                click.echo(click.style(error_msg, fg="yellow"))

            return

        # This will break if the response doesn't contain content length
        task_id = self.progress_bar.add(
            filename=filename,
            total=int(response.info().get("Content-length")),
        )

        with open(path, "wb") as dest_file:
            for data in iter(partial(response.read, self.chunk_size), b""):
                # parent has quit, die
                if not self.__active:
                    return

                dest_file.write(data)
                self.progress_bar.update(progress_bar_id=task_id, step=len(data))

        self.log.info("File %s downloaded successfully", dest_file)
        self.progress_bar.close(progress_bar_id=task_id)

    def download(self, urls: Iterable[str], dest_dir: str):
        """
        Download multuple files to the given directory

        Args:
            urls: list of urls from which files are to be downloaded
            dest_dir: downloaded files will end up here
        """
        self.__active = True

        # https://stackoverflow.com/q/29177490
        #
        # The threads created by ThreadPoolExecutor are daemon
        # Normally, when running as CLI, the parent dies and so do the threads
        #
        # When this is run inside a notebook as a magic command, this doesn't happen
        #   which means that threads are
        #    - still downloading files
        #    - continuously updating the progress bars

        try:
            with ThreadPoolExecutor(max_workers=self.njobs) as pool:
                for url in urls:
                    filename = urlparse(url).path.split("/")[-1]
                    dest_path = os.path.join(dest_dir, filename)
                    if os.name == "nt":
                        # TODO: Adding temperory hack for making donwload works
                        self.copy_url(filename, url, dest_path)
                        continue
                    pool.submit(self.copy_url, filename, url, dest_path)
        except KeyboardInterrupt:
            # the thread will read this
            self.__active = False
            raise
