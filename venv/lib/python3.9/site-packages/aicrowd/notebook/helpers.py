import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import unquote

import requests

from aicrowd.notebook import exceptions
from aicrowd.utils.jupyter import (
    is_google_colab_env,
    is_ipykernel_present,
    mount_google_drive,
)

log = logging.getLogger()


def read_notebook(file_path: str) -> Dict[str, Any]:
    log.debug("Reading notebook from: %s", file_path)
    try:
        with open(file_path) as fp:
            nb = json.load(fp)
    except UnicodeDecodeError:
        with open(file_path, encoding="utf-8") as fp:
            nb = json.load(fp)
    return nb


def write_notebook(file_path: str, nb: Dict[str, Any]):
    log.debug("Writing notebook to: %s", file_path)
    with open(file_path, "w") as fp:
        json.dump(nb, fp)


def delete_expressions_from_notebook(
    expressions: List[str], file_path: str
) -> Dict[str, Any]:
    """
    Delete code lines from the notebook that match the regular expressions

    Args:
        expressions: List of regular expressions to match
        file_path: Path to the notebook
    """
    nb = read_notebook(file_path)

    # TODO: Duplicated code, refactor
    for _cell in nb["cells"]:
        # Match the lines only in code blocks
        if _cell["cell_type"] != "code":
            continue
        source_code = []
        for _code_line in _cell["source"]:
            matched = False
            for _expr in expressions:
                if re.search(_expr, _code_line):
                    log.debug("Matched line to remove: %s", _code_line)
                    matched = True
                    break
            if not matched:
                source_code.append(_code_line)
        _cell["source"] = source_code
    return nb


def convert_timestamp_to_epoch(timestamp: str) -> float:
    pattern = "%Y-%m-%dT%H:%M:%S.%fZ"
    return time.mktime(time.strptime(timestamp, pattern))


def get_default_jupyter_api_session_host() -> str:
    if is_google_colab_env():
        return "http://172.28.0.2:9000"
    return "http://127.0.0.1:8888"


def get_latest_jupyter_session() -> Dict[str, Any]:
    proxies = {"http": None, "https": None}
    (
        auto_detected_jupyter_host,
        auto_detected_jupyter_token,
    ) = get_jupyter_server_endpoint()
    jupyter_notebook_host = os.getenv("JUPYTER_NB_HOST", auto_detected_jupyter_host)
    log.debug("Using jupyter server at %s", jupyter_notebook_host)

    response = requests.get(
        os.path.join(jupyter_notebook_host, "api/sessions"),
        proxies=proxies,
        headers={"Authorization": f"token {auto_detected_jupyter_token}"},
    )
    if not response.ok:
        log.debug("Failed to query jupyter session API")
        raise exceptions.InvalidJupyterResponse(
            f"Got invalid response from Jupyter: {response.text}"
        )
    log.debug("Available sessions on jupyter are %s", response.text)

    sessions = response.json()
    if len(sessions) == 0:
        raise exceptions.NotebookNotFound(
            "No active notebook detected. Is your notebook kernel running?"
        )
    latest_session = sorted(
        sessions,
        reverse=True,
        key=lambda x: convert_timestamp_to_epoch(
            x.get("kernel", {}).get("last_activity", -1)
        ),
    )[0]
    return latest_session


def get_notebook_path():
    if is_google_colab_env():
        log.debug("Detected google colab environment")
        return get_colab_notebook_via_gdrive()
    else:
        return get_jupyter_notebook_path()


def get_notebook(notebook_path: str = None) -> Tuple[Union[Dict[str, Any], str], str]:
    """Get notebook JSON or notebook file path and, notebook name.

    If notebook path is provided then returns the notebook path along with the notebook name else based on the kernel
    environment, returns notebook and notebook name. If the environment is Google colaboratory and if the code is
    running inside kernel, fetch notebook from frontend else mount google drive and get notebook path. If the
    environment is Jupyter notebook or Jupyterlab, get notebook path from disk. If the code is running inside kernel,
    save the notebook else show warning message.

    Args:
        notebook_path: Path of the notebook

    Returns:
        notebook: A JSON dictionary containing notebook contents or Path to the notebook file
        notebook_name: Name of the notebook
    """
    if notebook_path:
        return notebook_path, os.path.basename(notebook_path)
    if is_google_colab_env():
        log.debug("Detected google colab environment")
        return get_colab_notebook()
    return get_saved_jupyter_notebook()


def get_colab_notebook():
    notebook, notebook_name = None, None
    if is_ipykernel_present():
        notebook, notebook_name = get_colab_notebook_via_kernel()
    else:
        show_warning_to_use_magic_command(reason="to skip mounting the drive")
    if notebook is None:
        notebook = get_colab_notebook_via_gdrive()
        notebook_name = os.path.basename(notebook)
    return notebook, notebook_name


def get_saved_jupyter_notebook():
    notebook_path = get_jupyter_notebook_path()
    notebook_name = os.path.basename(notebook_path)
    if is_ipykernel_present():
        save_jupyter_notebook(notebook_path)
    else:
        show_warning_to_use_magic_command(reason="to save the notebook automatically")
    return notebook_path, notebook_name


def get_colab_notebook_via_kernel(
    timeout_sec: int = 180,
) -> Union[Tuple[Dict[str, Any], str], Tuple[None, None]]:
    """
    Get colab notebook from colab frontend via messaging
    """
    _message = import_module("google.colab._message")
    notebook = _message.blocking_request("get_ipynb", timeout_sec=timeout_sec)
    if notebook is None:
        return None, None
    return notebook["ipynb"], notebook["ipynb"]["metadata"]["colab"]["name"]


def save_jupyter_notebook(file_path: str, timeout_sec: int = 5):
    """Save jupyter notebook

    Use kernel comms messaging to save the notebook and wait until the notebook is saved or timeout occurs.

    Args:
        file_path: Notebook file path to be saved
        timeout_sec: Time in seconds to wait until file is getting saved
    """
    ipython_display = import_module("IPython.display")
    log.debug("Saving notebook: %s", file_path)
    deadline = time.time() + timeout_sec
    start_md5 = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    ipython_display.display(
        ipython_display.Javascript("IPython.notebook.save_checkpoint();")
    )
    current_md5 = start_md5
    while (start_md5 == current_md5) and (time.time() < deadline):
        log.debug("Waiting for the notebook to finish saving")
        time.sleep(1)
        current_md5 = hashlib.md5(open(file_path, "rb").read()).hexdigest()


def get_jupyter_notebook_path():
    latest_session = get_latest_jupyter_session()
    server_info = get_jupyter_server_info()

    notebook_root_dir = server_info.get("notebook_dir")
    if notebook_root_dir is None:
        log.debug("Failed to load root directory from `notebook_dir` attribute")
        notebook_root_dir = server_info.get("root_dir")
    if notebook_root_dir is None:
        log.debug("Failed to load root directory from `root_dir` attribute")
        notebook_root_dir = os.getcwd()
    notebook_path = os.path.join(notebook_root_dir, latest_session["path"])
    if os.path.exists(notebook_path):
        return os.path.abspath(notebook_path)

    log.debug("Notebook not found at %s", notebook_path)
    raise exceptions.NotebookNotFound(
        "Could not locate the absolute path to the jupyter notebook"
    )


def get_colab_notebook_via_gdrive():
    mount_path = mount_google_drive()
    return os.path.join(
        mount_path,
        "MyDrive/Colab Notebooks",
        unquote(get_latest_jupyter_session()["name"]),
    )


def bundle_notebook(submission_dir: str, notebook_name: str = None):
    if is_google_colab_env():
        raise exceptions.FeatureNotReady("Google colab submissions are not ready yet")
    else:
        bundle_original_jupyter_notebook(submission_dir, notebook_name)


def bundle_original_jupyter_notebook(submission_dir: str, notebook_name: str = None):
    submission_dir = submission_dir.replace(".zip", "")
    if notebook_name is None:
        notebook_name = get_jupyter_notebook_path()
    shutil.copy(notebook_name, os.path.join(submission_dir, "original_notebook.ipynb"))


def get_runtime_language() -> Tuple[str, str]:
    kernel = get_latest_jupyter_session()["kernel"]["name"]
    if kernel == "ir":
        log.debug("Using IR kernel")
        return "r", kernel
    if kernel.startswith("python"):
        log.debug("Using python kernel")
        return "python", kernel


def write_aicrowd_config(submission_dir: str):
    language, kernel = get_runtime_language()
    config = {
        "language": language,
        "kernel": kernel,  # we do not use this value, it's added only for debugging
    }
    log.debug("Writing runtime config to aicrowd.yaml")
    with open(os.path.join(submission_dir, "aicrowd.json"), "w") as fp:
        json.dump(config, fp)


def get_kernel_from_language(language: str) -> str:
    if language.lower() == "python":
        if is_google_colab_env():
            return "python3"
        return "python"
    if language.lower() == "r":
        return "ir"
    raise exceptions.FeatureNotReady("Unsupported language")


def import_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        log.debug("Failed to load module %s", module_name)


def get_server_info_from_module(notebook_server: ModuleType) -> Dict[str, str]:
    session_list = list(notebook_server.list_running_servers())
    log.debug("Available sessions on jupyter are %s", session_list)
    if len(session_list) > 1:
        print("WARNING: Got more than 1 jupyter server, selecting the latest session")
    if len(session_list) == 0:
        raise exceptions.NotebookAppImportException(
            "No jupyter server found. Did you start your jupyter server?"
        )
    return session_list[-1]


def get_jupyter_notebook_server_info() -> Dict[str, str]:
    notebook_server = import_module("notebook.notebookapp")
    if notebook_server is None:
        raise exceptions.NotebookAppImportException("No jupyter notebook installed")
    return get_server_info_from_module(notebook_server)


def get_jupyter_lab_server_info() -> Dict[str, str]:
    jupyter_server = import_module("jupyter_server.serverapp")
    if jupyter_server is None:
        raise exceptions.NotebookAppImportException("No jupyter lab installed")
    return get_server_info_from_module(jupyter_server)


def get_jupyter_server_info() -> Dict[str, str]:
    """
    Returns the latest jupyter server info
    """
    try:
        return get_jupyter_lab_server_info()
    except exceptions.NotebookAppImportException:
        log.debug("Found no active kernels from jupyter lab")
        return get_jupyter_notebook_server_info()


def get_jupyter_server_endpoint() -> Tuple[str, str]:
    """
    Returns the jupyter server endpoint along with the token

    Returns:
        Jupyter server endpoint and token
    """
    server_info = get_jupyter_server_info()
    return (
        server_info.get("url", get_default_jupyter_api_session_host()),
        server_info.get("token", ""),
    )


def show_warning_to_use_magic_command(reason: str = "") -> None:
    """Show warning message to use `%aicrowd` magic command with complete code

    Some features won't work with !aicrowd. `%aicrowd` magic command can be used to save
    notebook inside jupyter notebook/jupyterLab environment and also to get notebook directly from frontend without
    mounting the drive in colab environment.

    Args:
        reason: A string which describes the reason to use `%aicrowd` command over `!aicrowd`
    """
    description = (
        f"`%aicrowd` magic command can be used to save the notebook inside jupyter notebook/jupyterLab environment "
        f"and also to get the notebook directly from the frontend without mounting the drive in colab environment. "
        f"You can use magic command {reason} and submit using the code below:\n "
    )
    code = "\n".join(["%load_ext aicrowd.magic", "%aicrowd " + " ".join(sys.argv[1:])])
    warnings.warn(description + code)
