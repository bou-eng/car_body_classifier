import os
import re
from typing import Any, Dict

from aicrowd.notebook.helpers import read_notebook, write_notebook


class TargetBlocks:
    """
    Markdown cell heading in the notebook and their phase mappings
    """

    INSTALL_START = "^# Install packages"
    PRE_PROCESS_START = "^# Define preprocessing code|^# Import packages"
    TRAIN_START = "^# Training phase"
    PREDICT_START = "^# Prediction phase"
    SUBMIT = "^# Submit to AIcrowd"
    INSTALL_END = PRE_PROCESS_START
    TRAIN_END = PREDICT_START
    PRE_PROCESS_END = TRAIN_START


def split_notebook(file_path: str, output_dir: str):
    """
    Split the submission notebook into install.ipynb, train.ipynb and
    predict.ipynb

    **Note:** API key scrubbing should be done before this

    Args:
        file_path: Path to the submission notebook
        output_dir: Directory to place the output files in
    """
    print("Collecting notebook...")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    write_installation_notebook(file_path, output_dir)
    write_training_notebook(file_path, output_dir)
    write_prediction_notebook(file_path, output_dir)


def is_not_empty_cell(cell: Dict[str, Any]) -> bool:
    """
    Check for non empty cells

    Args:
        cell: Jupyter cell

    Returns:
        True of the cell is not empty
    """
    return len(cell["source"]) > 0


def extraction_direction(status: bool, remove: bool) -> bool:
    if remove:
        return not status
    return status


def extract_partial_notebook(
    nb: Dict[str, Any], start_block: str, end_block: str, remove: bool = False
) -> Dict[str, Any]:
    """
    Collects or removes cells blocks withing start and end headers

    Args:
        nb: Input notebook
        start_block: Regular expression for the starting block
        end_block: Regular expression for the ending block
        remove: Determines if

    Examples:
        To extract the notebook cells within `start_block` and `end_block`
        >>>extract_partial_notebook(nb, start_block="^# Heading 1", end_block="^# Heading 2")

        To remove the notebook cells within similar start and end blocks, use
        `remove=True`

    Returns:
        Notebook with cells with/without between the start and end blocks
    """
    target_cells = []
    start_copy = False
    for _cell in nb["cells"]:
        if is_not_empty_cell(_cell):
            if start_copy or re.search(start_block, _cell["source"][0]):
                start_copy = True
            if re.search(end_block, _cell["source"][0]):
                if not remove:
                    break
                start_copy = False
            if extraction_direction(start_copy, remove):
                target_cells.append(_cell)
    nb["cells"] = target_cells
    return nb


def write_installation_notebook(file_path: str, output_dir: str):
    """
    Extracts and writes install.ipynb to the submission directory

    Args:
        file_path: Path to the submission notebook
        output_dir: Path to the submission directory
    """
    nb = read_notebook(file_path)
    partial_nb = extract_partial_notebook(
        nb, TargetBlocks.INSTALL_START, TargetBlocks.INSTALL_END
    )
    write_notebook(os.path.join(output_dir, "install.ipynb"), partial_nb)


def write_training_notebook(file_path: str, output_dir: str):
    """
    Extracts and writes train.ipynb to the submission directory

    Args:
        file_path: Path to the submission notebook
        output_dir: Path to the submission directory
    """
    nb = read_notebook(file_path)
    partial_nb = extract_partial_notebook(nb, ".*", TargetBlocks.TRAIN_END)
    partial_nb = extract_partial_notebook(
        partial_nb, TargetBlocks.INSTALL_START, TargetBlocks.INSTALL_END, remove=True
    )
    write_notebook(os.path.join(output_dir, "train.ipynb"), partial_nb)


def write_prediction_notebook(file_path: str, output_dir: str):
    """
    Extracts and writes predict.ipynb to the submission directory

    Args:
        file_path: Path to the submission notebook
        output_dir: Path to the submission directory
    """
    nb = read_notebook(file_path)
    partial_nb = extract_partial_notebook(nb, ".*", TargetBlocks.SUBMIT)
    partial_nb = extract_partial_notebook(
        partial_nb, TargetBlocks.INSTALL_START, TargetBlocks.INSTALL_END, remove=True
    )
    partial_nb = extract_partial_notebook(
        partial_nb, TargetBlocks.TRAIN_START, TargetBlocks.TRAIN_END, remove=True
    )
    write_notebook(os.path.join(output_dir, "predict.ipynb"), partial_nb)
