"""
Lightweight configuration utilities.

Provides a thin wrapper around YAML file loading and a helper to merge
command-line arguments into a configuration dictionary.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.  Returns an empty dict if the file
        is empty.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML content cannot be parsed or is not a mapping at the
        top level.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        logger.warning("YAML file %s is empty; returning empty dict.", path)
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a top-level mapping in {path}, got {type(data).__name__}"
        )

    logger.info("Loaded config from %s (%d keys)", path, len(data))
    return data


def merge_cli_args(
    config: Dict[str, Any],
    args: Any,
    override_keys: Optional[list] = None,
) -> Dict[str, Any]:
    """Merge command-line arguments into an existing config dict.

    Values from *args* override those in *config*.  Only attributes that are
    not ``None`` are merged (so CLI defaults of ``None`` will not overwrite
    YAML values).

    Parameters
    ----------
    config : dict
        Base configuration dictionary (typically loaded from YAML).
    args : argparse.Namespace or dict
        Parsed CLI arguments.  If an ``argparse.Namespace``, its ``__dict__``
        is used.
    override_keys : list of str, optional
        If provided, only these keys from *args* are considered.  Otherwise
        all non-None values are merged.

    Returns
    -------
    dict
        A new dictionary with the merged configuration.  The original
        *config* is not modified.
    """
    merged = copy.deepcopy(config)

    if hasattr(args, "__dict__"):
        arg_dict = vars(args)
    elif isinstance(args, dict):
        arg_dict = args
    else:
        raise TypeError(
            f"args must be an argparse.Namespace or dict, got {type(args).__name__}"
        )

    keys_to_merge = override_keys if override_keys is not None else list(arg_dict.keys())

    for key in keys_to_merge:
        value = arg_dict.get(key)
        if value is not None:
            if key in merged:
                logger.debug("CLI override: %s = %r (was %r)", key, value, merged[key])
            else:
                logger.debug("CLI new key:  %s = %r", key, value)
            merged[key] = value

    return merged


def load_agent_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an agent config YAML with ``_base_`` inheritance and ``mode`` resolution.

    1. Load the YAML file.
    2. If a ``_base_`` key is present, load the base YAML first and
       deep-merge the task config on top.
    3. If a ``mode`` key is present and a sub-dict with that name exists
       (e.g. ``outdoor:``), flatten the selected mode's params into the
       top level and remove all mode sub-dicts.

    Returns
    -------
    dict
        Flat configuration dictionary ready to be unpacked as ``**kwargs``.
    """
    config = load_yaml(path)

    base_key = config.pop("_base_", None)
    if base_key:
        base_path = Path(path).parent / base_key
        base_config = load_yaml(base_path)
        base_config.update(config)
        config = base_config

    mode = config.get("mode")
    if mode and isinstance(mode, str):
        mode_dicts = {}
        for key in list(config.keys()):
            if isinstance(config[key], dict) and key != "mode":
                mode_dicts[key] = config.pop(key)

        if mode in mode_dicts:
            config.update(mode_dicts[mode])
            logger.info("Resolved mode '%s' with %d params", mode, len(mode_dicts[mode]))
        elif mode_dicts:
            logger.warning(
                "mode='%s' but no matching sub-dict found (available: %s)",
                mode,
                list(mode_dicts.keys()),
            )

    return config
