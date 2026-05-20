import json
import os

from aicrowd.notebook.helpers import delete_expressions_from_notebook

DEFAULT_CLEAN_EXPRESSIONS = [
    "API_KEY(\s+)?=(\s+)?[\"'].+[\"']",
    "^!aicrowd ",
]


def clean_notebook(file_path: str, output_path: str = None):
    """
    Remove sensitive data from the notebook and write the cleaned notebook
    in place

    Args:
        file_path: Path to the notebook
        output_path: Path to write the cleaned notebook to
    """
    print("Scrubbing API keys from the notebook...")
    clean_expressions = os.getenv("AICROWD_CLEAN_EXPRESSIONS")
    if clean_expressions is None:
        clean_expressions = DEFAULT_CLEAN_EXPRESSIONS
    else:
        clean_expressions = json.loads(clean_expressions)

    cleaned_nb = delete_expressions_from_notebook(clean_expressions, file_path)

    if output_path is None:
        output_path = file_path
    with open(output_path, "w") as fp:
        json.dump(cleaned_nb, fp)
