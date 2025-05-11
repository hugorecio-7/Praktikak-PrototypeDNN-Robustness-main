import sys

def is_local_env():
    is_notebook    = __name__ == '__main__' and '__file__' in globals()
    is_interactive = hasattr(sys, 'ps1')
    return is_notebook or is_interactive


def reload_module(module):
    from importlib import reload
    return reload(module)

