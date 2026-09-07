import * as vscode from "vscode";
import * as cp from "child_process";

interface Config {
  executablePath: string;
  defaultTimeout: number;
  model: string;
}

function getConfig(): Config {
  const cfg = vscode.workspace.getConfiguration("godspeed");
  return {
    executablePath: cfg.get<string>("executablePath", "godspeed"),
    defaultTimeout: cfg.get<number>("defaultTimeout", 0),
    model: cfg.get<string>("model", ""),
  };
}

function workspaceRoot(): string {
  const folders = vscode.workspace.workspaceFolders;
  return folders && folders.length > 0 ? folders[0].uri.fsPath : process.cwd();
}

function selectedText(editor: vscode.TextEditor): string {
  return editor.selection.isEmpty ? "" : editor.document.getText(editor.selection);
}

function buildArgs(task: string, config: Config, extra: string[] = []): string[] {
  const args: string[] = ["run", task];
  if (config.model) { args.push("--model", config.model); }
  if (config.defaultTimeout > 0) { args.push("--timeout", String(config.defaultTimeout)); }
  args.push("--json-output", "--auto-approve", "reads", "--project-dir", workspaceRoot());
  args.push(...extra);
  return args;
}

function streamCommand(
  channel: vscode.OutputChannel, label: string, executable: string, args: string[]
): Thenable<number> {
  return new Promise<number>((resolve, reject) => {
    channel.clear();
    channel.show(true);
    channel.appendLine(`[${label}] Starting: ${executable} ${args.join(" ")}\n`);
    const proc = cp.spawn(executable, args, {
      cwd: workspaceRoot(), shell: process.platform === "win32",
    });
    proc.stdout?.on("data", (d: Buffer) => channel.append(d.toString()));
    proc.stderr?.on("data", (d: Buffer) => channel.append(d.toString()));
    proc.on("error", (err: Error) => {
      channel.appendLine(`\n[${label}] Process error: ${err.message}`);
      reject(err);
    });
    proc.on("close", (code: number | null) => {
      const exitCode = code ?? 1;
      channel.appendLine(`\n[${label}] Exit code: ${exitCode}`);
      channel.appendLine(exitCode === 0 ? `[${label}] Done.` : `[${label}] Failed (exit ${exitCode}).`);
      resolve(exitCode);
    });
  });
}

function gitDiff(): string {
  try {
    return cp.execSync("git diff", {
      cwd: workspaceRoot(), encoding: "utf-8", timeout: 10_000,
    }).trim();
  } catch { return ""; }
}

const EXIT_LABELS: Record<number, string> = {
  1: "Tool error", 2: "Max iterations reached", 3: "Cost budget exceeded",
  4: "LLM provider failure", 5: "Invalid input", 6: "Timeout",
};

function notifyError(prefix: string, err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err);
  vscode.window.showErrorMessage(`${prefix}: ${msg}`);
}

export function activate(context: vscode.ExtensionContext): void {
  const channel = vscode.window.createOutputChannel("Godspeed");

  const runTaskCmd = vscode.commands.registerCommand("godspeed.runTask", async () => {
    const config = getConfig();
    let task = "";
    const editor = vscode.window.activeTextEditor;
    if (editor) { task = selectedText(editor); }
    if (!task) {
      const input = await vscode.window.showInputBox({
        prompt: "Enter a task for Godspeed",
        placeHolder: "e.g. Fix the failing test in test_auth.py",
      });
      if (!input) { return; }
      task = input;
    }
    try {
      const exitCode = await streamCommand(channel, "runTask", config.executablePath, buildArgs(task, config));
      if (exitCode !== 0) {
        const label = EXIT_LABELS[exitCode] ?? "Unknown error";
        vscode.window.showWarningMessage(`Godspeed runTask failed: ${label} (exit ${exitCode})`);
      }
    } catch (err) { notifyError("Godspeed runTask error", err); }
  });

  const explainSelectionCmd = vscode.commands.registerCommand(
    "godspeed.explainSelection", async () => {
      const config = getConfig();
      const editor = vscode.window.activeTextEditor;
      if (!editor) { vscode.window.showWarningMessage("No active editor."); return; }
      const selection = selectedText(editor);
      if (!selection) {
        vscode.window.showWarningMessage("Select some code first, then run Godspeed: Explain Selection.");
        return;
      }
      const task = `Explain the following code:\n\n\`\`\`\n${selection}\n\`\`\``;
      try {
        await streamCommand(channel, "explainSelection", config.executablePath, buildArgs(task, config));
      } catch (err) { notifyError("Godspeed explainSelection error", err); }
    }
  );

  const reviewDiffCmd = vscode.commands.registerCommand("godspeed.reviewDiff", async () => {
    const config = getConfig();
    const diff = gitDiff();
    if (!diff) { vscode.window.showInformationMessage("No unstaged changes found (git diff is empty)."); return; }
    const task = `Review the following git diff for bugs, security issues, and code quality:\n\n\`\`\`diff\n${diff}\n\`\`\``;
    try {
      await streamCommand(channel, "reviewDiff", config.executablePath, buildArgs(task, config));
    } catch (err) { notifyError("Godspeed reviewDiff error", err); }
  });

  const resumeCmd = vscode.commands.registerCommand("godspeed.resume", () => {
    const config = getConfig();
    const terminal = vscode.window.createTerminal({ name: "Godspeed (resume)", cwd: workspaceRoot() });
    terminal.sendText(`${config.executablePath} --continue`);
    terminal.show();
  });

  context.subscriptions.push(runTaskCmd, explainSelectionCmd, reviewDiffCmd, resumeCmd, channel);
}

export function deactivate(): void { /* No cleanup needed. */ }
