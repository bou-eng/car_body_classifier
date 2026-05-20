"""
%aicrowd magic command
"""
# this will never get called unless you are using ipython (or jupyter)
# since they will already have IPython installed, there's no need to add this as a dependency
# pylint: disable=E0401
from IPython.core.magic import Magics, line_magic, magics_class

# pylint: enable=E0401
from aicrowd.cli import cli


@magics_class
class AIcrowdMagics(Magics):
    """AIcrowd magic commands"""

    @line_magic
    def aicrowd(self, line: str):
        """Utility function to expose CLI commands as line magic commands

        Args:
            line: Text written next to the command
        """
        try:
            cli.main(line.split(" "), "%aicrowd")
        except Exception as e:
            print("An error occured:", e)
        except SystemExit:
            # Captures sys.exit from click
            pass


def load_ipython_extension(ipython):
    """Register our magic commands with ipython kernel

    The magic commands will be available as an extension amd can be loaded
    using

        %load_ext aicrowd.magic
    """
    ipython.register_magics(AIcrowdMagics)
