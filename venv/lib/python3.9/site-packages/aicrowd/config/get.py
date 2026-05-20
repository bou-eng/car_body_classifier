from aicrowd.contexts.context import ConfigContext
from aicrowd.config.exceptions import ConfigKeyNotFound
from aicrowd.errors import INVALID_PARAMETER


def config_get(key: str, config_ctx: ConfigContext = ConfigContext()):
    config = config_ctx.config._settings
    try:
        for key_path in key.split("."):
            config = config[key_path]
    except KeyError:
        raise ConfigKeyNotFound(
            f"Unknown config key: {key}", exit_code=INVALID_PARAMETER
        )

    return config
