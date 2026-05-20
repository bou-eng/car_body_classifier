"""
Submission related tasks
"""
import json
import logging
import os
import re
import shutil
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests
from git import Remote, Repo
from git.exc import GitError
from requests_toolbelt import MultipartEncoderMonitor
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from slugify import slugify

# pylint: disable=unused-import
from aicrowd.challenge.helpers import must_get_api_key, parse_cli_challenge

# pylint: enable=unused-import
from aicrowd.constants import AICROWD_GITLAB_HOST, RAILS_HOST
from aicrowd.errors import INVALID_CHALLENGE_REPO
from aicrowd.submission.exceptions import (
    ChallengeNotFoundException,
    SubmissionException,
    SubmissionFileException,
    SubmissionGitException,
    SubmissionUploadException,
)
from aicrowd.utils.git import ls_remote
from aicrowd.utils.jupyter import is_jupyter
from aicrowd.utils.utils import zip_fs_path
from aicrowd_api.challenge import check_registration_status
from aicrowd_api.submission import (
    create_rails_submission,
    get_submission_upload_details,
)


def calculate_min_table_width(table: Table):
    """
    Calculates minumum width needed to display table text properly

    Args:
        table: rich table
    """
    console = Console(width=200)
    width = (
        sum(table._calculate_column_widths(console, console.options))
        + table._extra_width
    )
    term_width = shutil.get_terminal_size().columns
    return max(width, term_width)


def get_upload_details(challenge_slug: str, api_key: str) -> Tuple[dict, bool]:
    """
    Contacts AIcrowd website for getting presigned url for uploading

    Args:
        challenge_slug: challenge slug
        api_key: AIcrowd API Key

    Returns:
        the data and whether the request was successful or not
    """
    log = logging.getLogger()

    r = get_submission_upload_details(api_key, challenge_slug)

    # temporary hack until /api stops redirecting
    if r.status_code // 100 == 3:
        redirected_to = r.headers["Location"]
        expected_redirect = f"https://{RAILS_HOST}"

        if redirected_to.startswith(expected_redirect):
            redirected_to = redirected_to[len(expected_redirect) :]
        else:
            log.error("Unexpected redirect location: %s", redirected_to)
            # got redirected to a weird place
            raise ChallengeNotFoundException(
                "Something went wrong with fetching challenge details"
            )

        challenge_problem = re.match(
            r"/challenges/([^/]*)/problems/([^/]*)/.*", redirected_to
        ).groups()
        logging.info("[metachallenge?] Got redirected to %s", redirected_to)

        # inform caller about meta_challenge
        return {
            "meta_challenge": True,
            "meta_challenge_id": challenge_problem[0],
            "challenge_id": challenge_problem[1],
        }, True

    try:
        resp = r.json()

        if not r.ok or not resp.get("success"):
            log.error(
                "Request to API failed\nReason: %s\nMessage: %s", r.reason, r.text
            )
            return resp, False

        return resp.get("data"), True
    except json.JSONDecodeError as e:
        log.error("Error while extracting details from API request: %s", e)
        raise ChallengeNotFoundException(
            "There was some error in contacting AIcrowd servers",
            "Please run this command with -v or -vv or .. -vvvvv to get more details",
        ) from e


class S3Uploader:
    """
    Upload files to s3 with progress bar
    """

    def __init__(self, host: str, fields: dict, file_path: str):
        """
        Args:
            host: s3 host to upload to
            fields: s3 related fields
            file_path: the file to upload
        """
        self.host = host
        self.fields = fields

        file_path = Path(file_path)

        self.file_name = file_path.name
        self.file_size = file_path.stat().st_size

        self.fields["key"] = self.fields["key"].replace("${filename}", self.file_name)
        self.fields["file"] = (self.file_name, file_path.open("rb"))

        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[file_name]}", justify="right"),
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
        self.task_id = self.progress.add_task(
            "upload", file_name=self.file_name, start=False, total=self.file_size
        )

    def track_progress(self, monitor: MultipartEncoderMonitor):
        """
        shows progress bar showing how much has been uploaded

        Args:
            monitor: requests_toolbelt Monitor
        """
        self.progress.update(self.task_id, completed=monitor.bytes_read, refresh=True)

    def upload(self):
        """
        upload file to s3
        """
        self.progress.start_task(self.task_id)
        m = MultipartEncoderMonitor.from_fields(
            self.fields, callback=self.track_progress
        )

        with self.progress:
            return (
                requests.post(
                    self.host, data=m, headers={"Content-Type": m.content_type}
                ),
                self.fields["key"],
            )


def upload_submission(host: str, fields: dict, file_path: str) -> Union[str, None]:
    """
    uploads a file to s3 using presigned url details

    Args:
        host: s3 host to upload to
        fields: s3 related keys
        file_path: the file to be uploaded

    Returns:
        whether it was successful or not
    """
    log = logging.getLogger()

    r, s3_key = S3Uploader(host, fields, file_path).upload()

    if not r.ok:
        log.error(
            "Couldn't upload file to s3\nReason: %s\nMessage: %s", r.reason, r.text
        )
        return None

    return s3_key


def notify_rails_upload(
    challenge_slug: str,
    submitted_url: str,
    api_key: str,
    description: str,
    problem_slug: Optional[str] = None,
    payload_extra_args: dict = {},
) -> dict:
    """
    notify rails about the uploaded file on s3

    Args:
        challenge_slug: challenge slug
        submitted_url: the url to which the submitted file was uploaded to
        description: submission description
        problem_slug: Used when submitting to a meta challenge

    Returns:
        submission details from AIcrowd API
    """
    log = logging.getLogger()

    payload: Dict[str, Any]
    if problem_slug is None:
        payload = {"challenge_id": challenge_slug}
    else:
        payload = {"meta_challenge_id": challenge_slug, "challenge_id": problem_slug}

    payload.update(
        {
            "submission": {"description": description},
            "submission_files": [{"submission_file_s3_key": submitted_url}],
        }
    )

    # only allow these extra args in payload
    # also, map keys to what rails expects
    extra_args_mapper = {"track": "clef_run_type"}
    for extra_arg in payload_extra_args:
        if extra_arg in extra_args_mapper:
            logging.debug(
                "Adding extra payload arg %s=%s",
                extra_args_mapper[extra_arg],
                payload_extra_args[extra_arg],
            )
            payload["submission"][extra_args_mapper[extra_arg]] = payload_extra_args[
                extra_arg
            ]

    logging.debug("Constructed payload %s", payload)
    r = create_rails_submission(api_key, payload)

    if not r.ok:
        log.error("Request to API failed\nReason: %s\nMessage: %s", r.reason, r.text)
        raise SubmissionUploadException(
            f"Couldn't decode response from AIcrowd servers. {r.reason}", r.text
        )

    try:
        return r.json()
    except json.JSONDecodeError as e:
        log.error("Couldn't json-decode rails response")
        raise SubmissionUploadException(
            "Couldn't decode response from AIcrowd servers"
        ) from e


def print_submission_links(challenge_slug: str, problem_slug: str, submission_id: int):
    """
    prints helpful links related to the submission

    Args:
        challenge_slug: challenge slug
        problem_slug: when submitting to a meta challenge
        submission_id: rails submission id
    """
    if submission_id is None:
        return

    if problem_slug:
        challenge_url = (
            f"https://{RAILS_HOST}/challenges/{challenge_slug}/problems/{problem_slug}"
        )
    else:
        challenge_url = f"https://{RAILS_HOST}/challenges/{challenge_slug}"

    submission_base_url = f"{challenge_url}/submissions"

    table = Table(title="Important links", show_header=False, leading=1, box=box.SQUARE)
    table.add_column(justify="right")
    table.add_column(overflow="fold")

    table.add_row("This submission", f"{submission_base_url}/{submission_id}")
    table.add_row("All submissions", f"{submission_base_url}?my_submissions=true")
    table.add_row("Leaderboard", f"{challenge_url}/leaderboards")
    table.add_row(
        "Discussion forum", f"https://discourse.aicrowd.com/c/{challenge_slug}"
    )
    table.add_row("Challenge page", f"{challenge_url}")

    width = calculate_min_table_width(table)

    console = Console(width=width)
    table.min_width = width
    console.print(Panel("[bold]Successfully submitted!"), justify="center")
    console.print(table)


def submit_file(
    challenge_slug: str,
    file_path: str,
    description: str,
    api_key: str,
    print_links: bool,
    payload_extra_args: dict = {},
) -> dict:
    """
    Submits a file given it's path and challenge_slug with given description

    Args:
        challenge_slug: challenge slug
        file_path: path to the file to be uploaded
        description: description for the submission
        api_key: AIcrowd API Key
        print_links: should helpful links be printed?

    Returns:
        a message from AIcrowd API
    """
    log = logging.getLogger()
    problem_slug = None

    if not Path(file_path).is_file():
        raise SubmissionFileException(f"Bad file {file_path}")

    s3_presigned_details, success = get_upload_details(challenge_slug, api_key)

    if s3_presigned_details.get("meta_challenge"):
        challenge_slug = s3_presigned_details.get("meta_challenge_id")
        problem_slug = s3_presigned_details.get("challenge_id")

        s3_presigned_details, success = get_upload_details(challenge_slug, api_key)

    if not success:
        log.error(
            "Error in getting presigned url for s3 upload: %s",
            s3_presigned_details.get("message"),
        )
        raise SubmissionUploadException(
            "Something went wrong while uploading your submission",
            s3_presigned_details.get("message"),
        )

    s3_key = upload_submission(
        s3_presigned_details["url"], s3_presigned_details["fields"], file_path
    )
    if s3_key is None:
        raise SubmissionUploadException(
            "Couldn't submit file. Please recheck the files and details provided"
        )

    response = notify_rails_upload(
        challenge_slug, s3_key, api_key, description, problem_slug, payload_extra_args
    )
    if not response.get("success"):
        log.error("Couldn't notify AIcrowd servers about uploaded submission")
        raise SubmissionUploadException(
            response.get(
                "message", "Something went wrong while contacting AIcrowd servers"
            )
        )

    if print_links:
        print_submission_links(
            challenge_slug, problem_slug, response.get("data", {}).get("submission_id")
        )

    return response.get("data")


def find_aicrowd_git_remote(repo: Repo) -> Remote:
    """
    Finds the appropriate git remote to push the submission to

    Checks whether `aicrowd` exists. then, finds any remote which points to AIcrowd gitlab
    """
    # check if `aicrowd` remote exists
    for remote in repo.remotes:
        if remote.name == "aicrowd":
            return remote

    # return the first remote with aicrowd gitlab as remote
    for remote in repo.remotes:
        if next(filter(lambda url: AICROWD_GITLAB_HOST in url, remote.urls), None):
            return remote

    raise SubmissionUploadException(
        "No git remote found",
        "Please re-create the repository with `aicrowd challenge init`",
        INVALID_CHALLENGE_REPO,
    )


def generate_submission_tag(repo: Repo, submission_description: str) -> str:
    """
    Generates a submission tag

    checks existing tags and creates a new one

    Args:
        repo: the git repo object for this challenge
        submission_description: description for this submission

    Returns:
        the (hopefully) non-conflicting generated tag
    """
    # TODO: what if the next tag exists on the remote but not here?

    tag_prefix = "submission-"

    base_tag = slugify(
        submission_description
        if submission_description.startswith(tag_prefix)
        else f"{tag_prefix}{submission_description}",
        regex_pattern=r"[^-a-z0-9.]+",
    )

    tag_already_exists = False

    for tag in repo.tags:
        if tag.name == base_tag:
            tag_already_exists = True
            break

    if not tag_already_exists:
        return base_tag

    largest_version = -1
    cli_tag_re = re.compile(f"^{base_tag}-v" + r"\d+$")

    for tag in repo.tags:
        if cli_tag_re.match(tag.name):
            tag_version = int(tag.name.split("-v")[1])
            if tag_version > largest_version:
                largest_version = tag_version

    return f"{base_tag}-v{largest_version+1}"


def submit_git(
    repo: Repo,
    submission_message: str,
    submission_tag: Optional[str] = None,
    print_links: bool = False,
):
    """
    Git based submissions

    the source code is committed and a tag starting with 'submission-' is pushed

    Args:
        repo: the git repo object for this challenge
        submission_message: this will be used as the commit message
        submission_tag: this will be used to name the tag
    """
    remote = find_aicrowd_git_remote(repo)

    if submission_tag is None:
        submission_tag = generate_submission_tag(repo, submission_message)
    if not submission_tag.startswith("submission-"):
        submission_tag = f"submission-{submission_tag}"

    if repo.is_dirty() or len(repo.untracked_files) > 0:
        # add everything from the repo root
        # assumes that gitignore has been set properly
        try:
            repo.git.add(all=True)
            repo.index.commit(submission_message)
        except GitError as e:
            raise SubmissionGitException(
                str(e),
                "Couldn't commit files. Is the git working tree in an unstable state?",
            ) from e

    try:
        repo.create_tag(submission_tag)
    except GitError as e:
        raise SubmissionGitException(str(e), "Couldn't create git tag") from e

    try:
        # this is needed to push the objects in case it is a new repo
        remote.push()
        # this actually pushes the tag
        remote.push(submission_tag)

        remote_url = next(remote.urls)

        # check if submission was successfully pushed
        assert f"refs/tags/{submission_tag}" in ls_remote(
            remote_url
        ), "Submission push failed, tag not found in the remote"
    except (GitError, AssertionError) as e:
        raise SubmissionGitException(
            str(e),
            "Couldn't push local changes. Has the remote repository been updated?",
        ) from e

    # not pointing to the exact issue, but to the issue list page
    # to get the exact issue, we'll need gitlab API key
    if print_links:
        gitlab_repo = remote_url.split(AICROWD_GITLAB_HOST)[1][1:]
        if gitlab_repo.endswith(".git"):
            gitlab_repo = gitlab_repo[:-4]

        gitlab_issues_url = f"https://{AICROWD_GITLAB_HOST}/{gitlab_repo}/-/issues"

        print(f"Check your submission status at {gitlab_issues_url}")


def verify_aicrowd_json():
    aicrowd_json = Path("aicrowd.json")
    if not aicrowd_json.is_file():
        raise SubmissionException(
            "aicrowd.json doesn't exist",
            "All git submissions should have the `aicrowd.json` file. "
            "Please clone the starter kit again",
        )

    try:
        aicrowd_json_dict = json.loads(aicrowd_json.read_text())
    except JSONDecodeError as e:
        raise SubmissionException(
            "aicrowd.json is not a valid json file",
            "Please clone the starter kit again",
        ) from e

    required_keys_types = [
        ("grader_id", str),
        ("challenge_id", str),
        ("authors", list),
        ("gpu", bool),
        ("description", str),
    ]
    for key, value_type in required_keys_types:
        if key not in aicrowd_json_dict or not isinstance(
            aicrowd_json_dict[key], value_type
        ):
            raise SubmissionException(
                f"Expected key {key} of type {value_type}",
                "Some keys are missing or have the wrong value type. "
                "Please refer to the original starter kit",
            )


def verify_participant_registration(api_key: str, challenge_id: int):
    """
    Checks whether participant has registered for the challenge or not
    """
    if not check_registration_status(api_key, challenge_id):
        raise SubmissionException(
            "You haven't registered for this challenge",
            "Please go to the challenge homepage and register",
        )


def verify_file_submission(api_key: str, challenge_id: int):
    """
    Checks for validity of file submissions
    """
    verify_participant_registration(api_key, challenge_id)


def verify_git_submission(api_key: str, challenge_id: int):
    """
    Checks for validity of git submission
    """
    verify_participant_registration(api_key, challenge_id)
    verify_aicrowd_json()


def zip_assets(assets_dir: str, target_zip_path: str):
    """
    Zip the files under assets directory

    Args:
        assets_dir: Directory containing submission assets
        target_zip_path: Path to zip file to add the entries to
    """
    console = Console()
    if not os.path.exists(assets_dir):
        os.mkdir(assets_dir)
        console.print("WARNING: Assets dir is empty", style="bold red")
    zip_fs_path(assets_dir, target_zip_path)


def delete_and_copy_dir(src: str, dst: str):
    """
    Delete if the src exists and copy files from src to dst

    Args:
        src: Source path
        dst: Destination path
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
