import * as vscode from 'vscode';
import { execFile } from 'child_process';
import { existsSync } from 'fs';
import { TextEncoder } from 'util';

function isExplicitPath(value: string): boolean {
  return value.includes('\\') || value.includes('/') || value.toLowerCase().endsWith('.exe');
}

async function getPythonPath(): Promise<string> {
  const pysiliconConfig = vscode.workspace.getConfiguration('pysilicon');
  const configured = pysiliconConfig.get<string>('pythonPath', '')?.trim() ?? '';

  if (!configured) {
    throw new Error(
      'PySilicon requires pysilicon.pythonPath to be set. Run "PySilicon: Select Python Interpreter" or set it in Settings.'
    );
  }

  if (!isExplicitPath(configured)) {
    throw new Error(
      `PySilicon pythonPath must be an explicit interpreter path. Received: ${configured}`
    );
  }

  if (!existsSync(configured)) {
    throw new Error(`PySilicon pythonPath does not exist: ${configured}`);
  }

  return configured;
}

async function pickAndStorePythonPath(): Promise<string | undefined> {
  const selection = await vscode.window.showOpenDialog({
    canSelectMany: false,
    canSelectFiles: true,
    canSelectFolders: false,
    openLabel: 'Use Interpreter',
    filters: process.platform === 'win32'
      ? { 'Python Executable': ['exe'] }
      : { 'All Files': ['*'] }
  });

  if (!selection || selection.length === 0) {
    return undefined;
  }

  const selectedPath = selection[0].fsPath;
  const config = vscode.workspace.getConfiguration('pysilicon');
  const target = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
    ? vscode.ConfigurationTarget.Workspace
    : vscode.ConfigurationTarget.Global;

  await config.update('pythonPath', selectedPath, target);
  void vscode.window.showInformationMessage(`PySilicon pythonPath set to: ${selectedPath}`);
  return selectedPath;
}

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand(
    'pysilicon.addHelloMethod',
    async (uri?: vscode.Uri) => {
      try {
        const activeEditor = vscode.window.activeTextEditor;
        const targetUri = uri ?? activeEditor?.document.uri;

        if (!targetUri) {
          vscode.window.showErrorMessage('No file selected. Open a file or run the command from Explorer.');
          return;
        }

        let text: string;
        if (activeEditor && activeEditor.document.uri.toString() === targetUri.toString()) {
          text = activeEditor.document.getText();
        } else {
          const data = await vscode.workspace.fs.readFile(targetUri);
          text = Buffer.from(data).toString('utf8');
        }

        let pythonPath: string;
        try {
          pythonPath = await getPythonPath();
        } catch (error) {
          const message = String(error);
          const pickAction = 'Select Python Interpreter';
          const selectedAction = await vscode.window.showErrorMessage(message, pickAction);

          if (selectedAction !== pickAction) {
            return;
          }

          const selectedPath = await pickAndStorePythonPath();
          if (!selectedPath) {
            return;
          }

          pythonPath = selectedPath;
        }

        const updated = await runPythonCreateHello(text, pythonPath);

        if (activeEditor && activeEditor.document.uri.toString() === targetUri.toString()) {
          const document = activeEditor.document;
          const lastLine = document.lineAt(document.lineCount - 1);
          const fullRange = new vscode.Range(0, 0, document.lineCount - 1, lastLine.range.end.character);
          await activeEditor.edit((editBuilder) => {
            editBuilder.replace(fullRange, updated);
          });
        } else {
          const enc = new TextEncoder().encode(updated);
          await vscode.workspace.fs.writeFile(targetUri, enc);
        }

        vscode.window.showInformationMessage('Added say_hello()!');
      } catch (err) {
        vscode.window.showErrorMessage(`Error: ${err}`);
      }
    }
  );

  context.subscriptions.push(disposable);

  const selectPythonPathDisposable = vscode.commands.registerCommand(
    'pysilicon.selectPythonPath',
    async () => {
      try {
        await pickAndStorePythonPath();
      } catch (error) {
        void vscode.window.showErrorMessage(`Failed to set PySilicon pythonPath: ${String(error)}`);
      }
    }
  );

  context.subscriptions.push(selectPythonPathDisposable);
}
function runPythonCreateHello(input: string, pythonPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = execFile(
      pythonPath,
      [
        "-c",
        `
import sys
from pysilicon.dev.create_hello import create_hello
print(create_hello(sys.stdin.read()))
        `
      ],
      { maxBuffer: 10 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          const stderrText = stderr?.toString().trim();
          const prefix = `Python execution failed using interpreter: ${pythonPath}`;
          if (stderrText) {
            reject(new Error(`${prefix}\n${error.message}\n${stderrText}`));
            return;
          }
          reject(new Error(`${prefix}\n${error.message}`));
          return;
        }
        resolve(stdout);
      }
    );

    if (proc.stdin) {
      proc.stdin.write(input);
      proc.stdin.end();
    } else {
      reject(new Error("Python process has no stdin"));
    }
  });
}


export function deactivate() {}