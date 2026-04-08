---
title: Installing the Python Package
parent: Installation
nav_order: 1
has_children: false
---

# Cloning and Installing the Python Package

## Cloning the repository

First clone the [GitHub repository](https://github.com/sdrangan/pysilicon) to your PC.

You can clone the repository with the command:

```bash
git clone https://github.com/sdrangan/pysilicon.git
```

Since the repository is frequently updated, you may need to pull the latest changes. To fetch and override any local changes:

```bash
git fetch origin
git reset --hard origin/main
```


## Installing the package

* First, create a virtual environment. The command below will
  create an environment named `env`,
  but any other name can be used.
  I usually run this command in the directory just outside `pysilicon`.

```bash
python -m venv env
```

  The command may take several minutes and may not indicate its progress.
  After completion, the virtual environment files will be in a directory named `env`.
  This directory may be large.

* Activate the virtual environment:

```bash
.\env\Scripts\Activate.ps1  # Windows PowerShell
.\env\Scripts\activate.bat  # Windows Command Prompt
source env/bin/activate     # macOS / Linux
```

  On Windows PowerShell, you may get the error:
  *"...Activate.ps1 is not digitally signed. The script will not execute on the system."*
  In this case, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

* The first time you activate the environment, install the package requirements:

```bash
(env) pip install -r path/to/requirements_loose.txt
```

  This installs the latest version of all packages. If you want the exact versions
  used when the package was created, run:

```bash
(env) pip install -r path/to/requirements.txt
```

* Next, install the `pysilicon` package in editable mode:

```bash
(env) pip install -e /path/to/pysilicon
```

  This step only needs to be done once per virtual environment.


## Creating a requirements file

If you update the package dependencies, you may need to regenerate the `requirements.txt` file:

```bash
python -m pip freeze > pysilicon/requirements.txt
```

The above `requirements.txt` will contain precise version numbers.
To create a version with version constraints stripped (Windows PowerShell):

```powershell
(Get-Content requirements.txt) -replace "[<>=~!].*","" | Set-Content requirements-loose.txt
```

On macOS / Linux:

```bash
sed 's/[<>=~!].*//' requirements.txt > requirements-loose.txt
```

After generating `requirements.txt`, you may want to edit it as follows:

* If you see a line like:

  ```
  pywin32==306
  ```

  Delete it — this package is only needed on Windows.

* If you see a line like:

  ```
  -e git+https://github.com/sdrangan/pysilicon.git@...#egg=utils
  ```

  Delete it — this line installs the `pysilicon` package directly from GitHub,
  but since you already installed it in editable mode, it is not needed.
