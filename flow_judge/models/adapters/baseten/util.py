def is_interactive() -> bool:
    """Check if the current environment is interactive.

    :return: True if the environment is interactive, False otherwise.
    :rtype: bool
    """
    import sys

    return sys.__stdin__.isatty()
