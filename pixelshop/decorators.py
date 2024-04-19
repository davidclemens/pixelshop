#!/usr/bin/env python3

"""
Decorators

The decorators module provides decorators.

Classes:

    none

Functions:

    none

Misc variables:

    none

(c) 2024 David Clemens
"""

import copy
from typing import Callable


def chainable(func: Callable):
    """
    The chainable decorator makes a method chainable without side effects.

    Arguments:
        func {Callable} -- The method to decorate
    """

    def wrapperChainable(*args, **kwargs):
        self, *args = args  # Extract self from *args
        copiedSelf = copy.deepcopy(self)  # Create deep copy
        returnValues = func(copiedSelf, *args, **kwargs)  # Call the wrapped function
        return returnValues

    return wrapperChainable


def autosave(suffix: str = ""):
    """
    The autosave decorator supresses autosave for all nested calls and performs
    an autosave with a specified suffix instead. It restores the autosave state
    afterwards.

    Keyword Arguments:
        suffix {str} -- The filename suffix to use for the decorated method
            (default: "")
    """

    def decorator(func: Callable):
        def wrapperAutosave(*args, **kwargs):
            self, *args = args  # Extract self from *args
            autoSaveState = self.autoSave  # Remember autoSave state
            self.autoSave = False  # Disable autoSave
            returnValues = func(self, *args, **kwargs)  # Call original function

            if isinstance(returnValues, tuple):
                self, *returnValues = returnValues
                flag = 1
            else:
                self = returnValues
                flag = 2

            self.autoSave = autoSaveState  # Reenable autoSave
            self._doAutoSave(appendedName=suffix)  # Perform autoSave

            if flag == 1:
                return self, *returnValues
            elif flag == 2:
                return self

        return wrapperAutosave

    return decorator
