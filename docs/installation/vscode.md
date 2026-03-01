---
title: Installing the VS Code Extension
parent: Installation
nav_order: 1
has_children: false
---

# VS Code Extension Setup

This guide describes how to set up the development environment for the PySilicon VS Code extension. These steps are only required for contributors working on the extension itself. Users of PySilicon do not need Node.js or the extension build environment.

---

## 1. Install prerequisites

You will need:

- Node.js (LTS recommended): https://nodejs.org
- Visual Studio Code: https://code.visualstudio.com

No Python virtual environment is required for the extension.

To validate the NodeJS installation, open a terminal (e.g., Powershell on Windows or command terminal on Unix / MacOS):

```bash
> node -v
v24.14.0
> npm -v
11.9.0
```

The precise version numbers may change.

If you are on Windows, you may get a permission denied.  In this case, open Powershell as administrator and run:

```bash
Set-ExecutionPolicy RemoteSigned
```



---

## 2. Navigate to the extension folder

From the root of the repository:

```
cd pysilicon/pysilicon-extension
```

This folder contains the extension’s `package.json`, `tsconfig.json`, and TypeScript source files.

---

## 3. Install Node dependencies

Run:

```
npm install
npm install --save-dev @types/node
```

This installs:

- TypeScript compiler
- VS Code type definitions
- Node type definitions (required for Buffer, process, etc.)

All dependencies are local to this folder. Nothing is installed globally.

---

## 4. Build the extension

To compile once:

```
npm run compile
```

To compile continuously while editing:

```
npm run watch
```

This generates JavaScript output in the `out/` directory.

---

## 5. Launch the Extension Development Host

Open the **root** PySilicon repository in VS Code:

```
pysilicon/
```

Then press:

```
F5
```

This opens a new VS Code window called **Extension Development Host** with the extension loaded.

---

## 6. Open the toy example workspace

In the Extension Development Host window:

1. File → Open Folder…
2. Select:

```
pysilicon/examples/vscodedemos/
```

3. Open `myclass.py`
4. Run the command:

```
PySilicon: Add Hello Method
```

The file should update in place.

---

## 7. Editing the extension

Edit files in:

```
pysilicon/pysilicon-extension/src/
```

If you are running `npm run watch`, TypeScript rebuilds automatically.  
Reload the Extension Development Host window with:

```
Ctrl+R
```

Changes take effect immediately.

---

## 8. Common commands

```
npm install          # install dependencies
npm run compile      # build once
npm run watch        # build continuously
F5                   # launch Extension Development Host
Ctrl+R               # reload extension in the host window
```