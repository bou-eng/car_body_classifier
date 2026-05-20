"""
Top level helpers
"""
import os
import requests
import semver
import subprocess
import sys
import uuid
from abc import ABC
from functools import wraps
from typing import Any

import click
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tqdm.auto import tqdm

from aicrowd import __version__
from aicrowd.errors import UNKNOWN_ERROR
from aicrowd.exceptions import CLIException
from aicrowd.utils.jupyter import is_jupyter


def is_subsequence(s1: str, s2: str) -> bool:
    """
    returns whether s1 is a subsequence of s2 or not

    Args:
        s1: probable subsequence of s2
        s2: string

    Returns:
        is s1 a subsequence of s2
    """
    i, j = 0, 0
    n1, n2 = len(s1), len(s2)

    while i < n1 and j < n2:
        # if matched, advance in both
        if s1[i] == s2[j]:
            i += 1

        # otherwise, try next char in s2
        j += 1

    # was s1 fully matched?
    return i == n1


DEPRECATED_HELP_NOTICE = " (DEPRECATED)"


class AliasedGroup(click.Group):
    """
    Click group allowing using prefix of command instead of the whole command
    """

    def get_command(self, ctx, cmd_name):
        """
        Returns a command of which the given word is a subsequence
        """
        rv = click.Group.get_command(self, ctx, cmd_name)

        if rv is not None:
            return rv

        matches = list(
            filter(
                lambda cmd_orig: is_subsequence(cmd_name, cmd_orig),
                self.list_commands(ctx),
            )
        )

        if not matches:
            return None

        if len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])

        ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")
        return None


class CommandWithExamples(click.Command):
    """
    Click command which extracts and properly displays usage examples
    """

    def format_help(self, ctx, formatter):
        """
        Modified --help output to also include examples
        """
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_examples(ctx, formatter)
        self.format_epilog(ctx, formatter)

    def format_help_text(self, ctx, formatter):
        """
        Don't print 'Examples' as a part of help
        """
        if self.help:
            formatter.write_paragraph()
            with formatter.indentation():
                help_text = self.help.split("Examples:")[0]
                if self.deprecated:
                    help_text += DEPRECATED_HELP_NOTICE
                formatter.write_text(help_text)
        elif self.deprecated:
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(DEPRECATED_HELP_NOTICE)

    def format_examples(self, _ctx, formatter):
        """
        Show examples separately, in their own section
        """
        try:
            examples_index = self.help.index("Examples:")
            formatter.write("\n")
            formatter.write(self.help[examples_index:])
            formatter.write("\n")
        except ValueError:
            # this command didn't provide examples
            pass


def zip_fs_path(fs_path: str, target_zip_path: str):
    fs_path = fs_path.replace(".zip", "")
    if not os.path.exists(fs_path):
        raise FileNotFoundError(f"The path {fs_path} does not exist")
    if (
        subprocess.run(f"zip -r '{target_zip_path}' '{fs_path}'", shell=True).returncode
        != 0
    ):
        raise Exception(f"Failed to zip {fs_path}")


class ProgressBar(ABC):
    def add(self, filename: str, total: int):
        raise NotImplementedError

    def update(self, progress_bar_id: Any, step: int):
        raise NotImplementedError

    def close(self, progress_bat_id: Any):
        raise NotImplementedError


class RichProgressBar(ProgressBar):
    def __init__(self):
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )
        self.progress.console.is_jupyter = is_jupyter()
        self._tasks = []

    def add(self, filename: str, total: int, **kwargs):
        task_id = self.progress.add_task(
            "download",
            filename=filename,
            total=total,
            start=True,
            refresh=True,
            **kwargs,
        )
        self._tasks.append(task_id)
        return task_id

    def update(self, progress_bar_id: TaskID, step: int):
        self.progress.update(progress_bar_id, advance=step, refresh=True)

    def close(self, progress_bar_id: TaskID):
        pass


class TqdmProgressBar(ProgressBar):
    def __init__(self):
        self.bars = {}

    def add(self, filename: str, total: int, **kwargs):
        bar_id = uuid.uuid4()
        self.bars[bar_id] = tqdm(
            desc=filename, total=total, unit="B", unit_scale=True, **kwargs
        )
        return bar_id

    def update(self, progress_bar_id: Any, step: int, **kwargs):
        self.bars[progress_bar_id].update(step, **kwargs)

    def close(self, progress_bar_id: str):
        self.bars[progress_bar_id].close()


def exception_handler(f):
    """
    try/except wrapper for CLI function calls
    """

    @wraps(f)
    def _exception_handler(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except CLIException as e:
            if not e.message:  # covers both "" and None
                e.message = (
                    "An unknown error occured. Please set verbosity to check full logs."
                )

            click.echo(str(e), err=True)
            sys.exit(e.exit_code)
        except Exception as e:
            click.echo(click.style("An unexpected error occured!", fg="red"), err=True)
            click.echo(
                f"{e}\n"
                + "To get more information, you can run this command with -v.\n"
                + "To increase level of verbosity, you can go upto -vvvvv",
                err=True,
            )
            sys.exit(UNKNOWN_ERROR)

    return _exception_handler


def check_for_latest_version():
    try:
        latest_version = requests.get(
            "https://pypi.org/pypi/aicrowd-cli/json", timeout=3
        ).json()["info"]["version"]
        if (
            semver.compare(
                __version__[1:],
                latest_version,
            )
            < 0
        ):
            click.echo(
                click.style(
                    f"You are using {__version__} but v{latest_version} is available."
                    + " Update for the latest features!\n"
                    + "pip install -U aicrowd-cli",
                    fg="yellow",
                ),
                err=True,
            )
    except:
        pass
