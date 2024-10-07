# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from termcolor import colored


def _flatten(args: tuple) -> str:
    """Flatten a tuple of arguments into a string joined by spaces.

    Args:
        args (tuple): Tuple of arguments to flatten.

    Returns:
        str: Flattened string.
    """
    return " ".join([str(arg) for arg in args])


def error(*args) -> None:
    print(colored(_flatten(args), "red"))


def info(*args) -> None:
    print(*args)


def warning(*args) -> None:
    print(colored(_flatten(args), "yellow"))


def alert(*args) -> None:
    print(colored(_flatten(args), "green"))