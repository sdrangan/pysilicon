"""
Packaged schema examples for pysilicon.

This sub-package ships the curated schema example files so they are available
via ``importlib.resources`` even in installed (wheel/sdist) environments.
Files can be enumerated with::

    from importlib import resources
    pkg = resources.files("pysilicon.examples")
    for resource in pkg.iterdir():
        ...
"""
