"""
For click contexts
"""

import click

from aicrowd.contexts import Challenge, CLIConfig


class ConfigContext:
    """
    Contains:
     - CLIConfig
    """

    def __init__(self):
        self.config: CLIConfig = CLIConfig()

    def __repr__(self):
        return "ConfigContext()"


class ChallengeContext:
    """
    Contains information about the challenge for cwd
    """

    def __init__(self):
        self.challenge: Challenge = Challenge()

    def __repr__(self):
        return "ChallengeContext()"


pass_config = click.make_pass_decorator(ConfigContext, ensure=True)
pass_challenge = click.make_pass_decorator(ChallengeContext, ensure=True)
