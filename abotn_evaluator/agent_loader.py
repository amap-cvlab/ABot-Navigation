"""
Dynamic agent and evaluator class loading utilities.

Provides helpers to import classes from dotted module paths at runtime,
validate that loaded agents expose the required interface, and detect
whether an agent implements the required interface methods.
"""

import importlib
import logging
from typing import Optional, Type

logger = logging.getLogger(__name__)


def load_class(module_path: str) -> Type:
    """Import and return a class from a ``package.module:ClassName`` path.

    Parameters
    ----------
    module_path : str
        Fully-qualified path in the form ``"package.module:ClassName"``.
        The colon separates the importable module from the attribute name
        within that module.

    Returns
    -------
    type
        The resolved class object.

    Raises
    ------
    ValueError
        If *module_path* does not contain exactly one ``:``.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class name is not found in the module.

    Examples
    --------
    >>> cls = load_class("agent_examples.point_goal_random:RandomPointGoalAgent")
    >>> agent = cls(config)
    """
    if ":" not in module_path:
        raise ValueError(
            f"module_path must use the 'package.module:ClassName' format. "
            f"Got: {module_path!r}"
        )

    parts = module_path.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"module_path must contain exactly one ':'. Got: {module_path!r}"
        )

    module_name, class_name = parts
    module_name = module_name.strip()
    class_name = class_name.strip()

    if not module_name or not class_name:
        raise ValueError(
            f"Both module and class name must be non-empty. "
            f"Got module={module_name!r}, class={class_name!r}"
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import module {module_name!r}. "
            f"Make sure the package is installed and the module path is correct. "
            f"Original error: {exc}"
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        available = [
            name for name in dir(module)
            if not name.startswith("_") and isinstance(getattr(module, name, None), type)
        ]
        raise AttributeError(
            f"Module {module_name!r} has no attribute {class_name!r}. "
            f"Available classes: {available}"
        ) from exc

    if not isinstance(cls, type):
        raise TypeError(
            f"{module_name}:{class_name} is not a class (got {type(cls).__name__})"
        )

    logger.info("Loaded class %s from %s", class_name, module_name)
    return cls
