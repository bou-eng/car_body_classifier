"""
Contains project version information
"""

import click


@click.command(name="version")
def version_command():
    """
    Prints AIcrowd CLI version
    """
    from aicrowd import __version__

    click.echo(click.style(str(__version__), bold=True))
