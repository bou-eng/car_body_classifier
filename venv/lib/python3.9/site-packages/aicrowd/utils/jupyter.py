import os
import subprocess

from rich.console import Console

from aicrowd.constants import JupyterConfigConstants


def is_ipykernel_present():
    """
    Check if process is running inside IPython kernel
    """
    try:
        # pylint: disable=import-error,no-name-in-module
        from IPython import get_ipython

        ipy_kernel = get_ipython()
        return ipy_kernel is not None
    except ImportError:
        return False


def get_ipython_object():
    # this will never succeed unless you are using ipython (or jupyter)
    # since they will already have IPython installed, there's no need to
    #   add this as a dependency

    from IPython import get_ipython  # pylint: disable=import-error

    return get_ipython()


def is_jupyter():
    """
    Tries to guess whether it's running inside jupyter notebook or not
    """
    # first, check if get_ipython() itself is defined
    try:
        ip = get_ipython_object()
        # now check if it's running as kernel
        return ip.has_trait("kernel")
    except:
        return False


def is_google_colab_env() -> bool:
    """
    Checks if the CLI is running on Google colab notebook.
    """
    try:
        import google.colab  # pylint: disable=import-error

        return True
    except ImportError:
        return False


def mount_google_drive(
    mount_path: str = "/content/drive",
    mount_reason: str = "Your Google Drive will be mounted to access the colab notebook",
) -> str:
    """Mount user's Google Drive to a path

    Args:
        mount_path: Path in local FS where the user's GDrive will be mounted
        mount_reason: Message displayed on the terminal when attempting to mount

    Returns:
        Mount path for user's GDrive
    """
    if os.path.exists(mount_path):
        return mount_path

    from google.colab import drive  # pylint: disable=import-error,no-name-in-module

    console = Console()
    console.print("Mounting Google Drive :floppy_disk:", style="bold blue")
    console.print(mount_reason)
    drive.mount(mount_path)
    return mount_path


def execute_command(command: str) -> int:
    """Capture the outputs from subprocess and send them to current process'
    stdout

    The output to subprocess' stdout will not be printed on the jupyter
    notebook. So we need to capture the output and print to current process'
    stdout.

    Args:
        command: Bash command to execute

    Returns:
        Exit code of the bash command
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    def print_output(handle):
        if handle is not None:
            output = handle.read()
            if output:
                print(output.decode().strip())

    while True:
        if process.poll() is not None:
            break
        print_output(process.stdout)
        print_output(process.stderr)
    rc = process.poll()
    return rc
