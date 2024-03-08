"""
General purpose functions.
"""
import inspect
from functools import wraps


def initializer(init_func):
    """ Automatically assigns named parameters of __init__ methods. """
    arg_names, _, _, defaults, _, keyword_only_defaults, _ = inspect.getfullargspec(init_func)
    arg_names = arg_names[1:]  # all argument names after self

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        if defaults is not None:
            for name, default in zip(reversed(arg_names), reversed(defaults)):
                setattr(self, name, default)

        if keyword_only_defaults is not None:
            for key, value in keyword_only_defaults.items():
                setattr(self, key, value)

        for name, arg in list(zip(arg_names, args)) + list(kwargs.items()):
            setattr(self, name, arg)

        init_func(self, *args, **kwargs)

    return wrapper
