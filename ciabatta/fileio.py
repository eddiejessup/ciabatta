"""
Functions related to interacting with the file-system.
"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)
from builtins import input

import os
import subprocess


def get_git_hash():
    """Return the shortened git SHA of the current commit."""
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).strip()


def makedirs_safe(dirname):
    """Make a directory, prompting the user if it already exists."""
    if os.path.isdir(dirname):
        s = input('%s exists, overwrite? (y/n) ' % dirname)
        if s != 'y':
            raise Exception
    else:
        os.makedirs(dirname)


def makedirs_soft(dirname):
    """Make a directory, if it doesn't already exist. Otherwise do nothing."""
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
