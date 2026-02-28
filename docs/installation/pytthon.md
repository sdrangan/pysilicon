---
title: Installing the Python Package
parent: Installation
nav_order: 1
has_children: false
---

# Cloning and Installing the Python packaage

## Cloning the repoistory
First clone [GitHub repository](https://github.com/sdrangan/pysilicon) to your PC.

You can clone the repository to your host PC with the command:
~~~bash
    git clone https://github.com/sdrangan/hwdesign.git
~~~
Since I am frequently updating material, you may need to reload the repository.  If you want to pull it and override local changes:
~~~bash
    git fetch origin
    git reset --hard origin/main
~~~


## Installing the package

*  First, create a virtual environment.  The command below will
    create an environment named `env`,
    but any other environment name can be used.
    In fact, since I have many virtual environments, I call it `pysilicon-venv`.
    I usually perform this command in the directory just outside `pysilicon`.

```bash
python -m venv env
```

    The command may take several minutes, and it may not indicate
    its progress. After completion, the virtual environment files will be in a
    directory `env`.  This directory may be large.

* Activate the virtual environment:

```bash
.\env\Scripts\Activate.ps1  [Windows powershell]
.\env\Scripts\activate.bat  [Windows command prompt]
source env/bin/activate [MAC/Linux]
```

    On Windows Powershell, you may get the error message
    *“...Activate.ps1 is not digitally signed. The script will not execute on the system.”*
    In this case, you will want to run:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
   
* The first time you activate the environment, install the
    package requirements:

```bash
(env) pip install -r path/to/requirements_loose.txt
```

    This command the latest version of all packages.  If you want the exact versions
    used when the package was created, run:

```bash
(env) pip install -r path/to/requirements.txt
```


*  Next, install the `pysilicon` package as editable:

```bash
(env) pip install -e /path/to/hwdesign
```

    This step also only needs to be done once for the virtual environment.


## Creating a requirements file
If you update the installation in the package, you may need to re-create the
`requirements.txt` file with:

```bash
python -m pip freeze > hwdesign/requirements.txt
```

The above `requirements.txt` will have the precise version numbers.
To create the version with the version numbers in Windows Powershell:

```bash
(Get-Content requirements.txt) -replace "[<>=~!].*","" | Set-Content requirements-loose.txt  
```

In MacOS / Linux:

```bash
sed 's/[<>=~!].*//' requirements.txt > requirements-loose.txt
```

If you do this on Windows, you should edit the file `requirements.txt` as follows:

* In `requirements.txt`, you may have a line like:

```
pywin32==306
```

  Delete this line since it is only needed for Windows.

* You may also find a line like:

```
-e git+https://github.com/sdrangan/pysilicon.git@...#egg=xilinxutils
```

The particular github address may be different and there may be a long version number.
This line installs the `pysilicon` package directly from github.  But, we do not need it.
So delete this line as well.


