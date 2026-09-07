# Godspeed Coding Agent — VS Code Extension

A minimal VS Code side-panel integration for the [Godspeed coding agent](https://github.com/t-timms/godspeed-coding-agent). Sends tasks from your editor to `godspeed run` and streams results into an Output Channel.

## Features

| Command | What it does |
|---------|-------------|
| **Godspeed: Run Task** | Send editor selection (or type a prompt) to `godspeed run` with `--json-output`. Streams stdout/stderr into the *Godspeed* output channel and surfaces exit codes. |
| **Godspeed: Explain Selection** | Wraps the current selection in an "explain this code" prompt and sends it to `godspeed run`. |
| **Godspeed: Review Diff** | Runs `git diff` in the workspace root, sends the result as a review-style prompt to `godspeed run`. |
| **Godspeed: Resume Session** | Opens an integrated terminal running `godspeed --continue` to resume the most recent conversation. |

### Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `godspeed.executablePath` | string | `"godspeed"` | Path to the godspeed binary. Use an absolute path if it is not on your `PATH`. |
| `godspeed.defaultTimeout` | integer | `0` | Wall-clock timeout in seconds for `godspeed run` (0 = no limit). Maps to `--timeout`. |
| `godspeed.model` | string | `""` | Model override passed via `--model`. Leave empty to use the default from `settings.yaml`. |

## Install from Source

This extension is not published to the VS Code Marketplace. Build and install locally:

```bash
cd ide/vscode
npm install          # or: npx tsc -p ./   (after manual @types/vscode + typescript install)
npx tsc -p .         # compile TypeScript → out/
```

Then in VS Code:
1. Open this folder (`ide/vscode`) with **File → Open Folder**.
2. Press `F5` to launch the Extension Development Host.

Or package and install:

```bash
npm install -g @vscode/vsce
vsce package         # produces godspeed-coding-agent-0.1.0.vsix
code --install-extension godspeed-coding-agent-0.1.0.vsix
```

**Prerequisites:** Godspeed must be installed and on your `PATH` (or set `godspeed.executablePath`).

## How It Works

Every command (except *Resume*) spawns `godspeed run "<task>" --json-output --auto-approve reads --project-dir <workspace>` via `child_process.spawn`. Stdout and stderr are streamed line-by-line into a VS Code Output Channel. When the process exits, the extension surfaces the exit code:

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | Tool error |
| 2 | Max iterations reached |
| 3 | Cost budget exceeded |
| 4 | LLM provider failure |
| 5 | Invalid input |
| 6 | Timeout |
| 130 | Interrupted (SIGINT) |

The *Resume* command opens an integrated terminal running `godspeed --continue`.

## Limitations

- **No inline diff preview.** Results appear in the Output Channel only; there is no side-by-side diff view or inline decoration.
- **No authentication management.** API keys must be configured via environment variables or `~/.godspeed/settings.yaml` before the extension can work.
- **No background task tracking.** Long-running tasks occupy the output channel; there is no task queue or progress indicator.
- **Single workspace.** Uses the first workspace folder as the project root. Multi-root workspaces are not supported.
- **Alpha quality.** This is an early integration scaffold, not a polished marketplace extension. Expect rough edges.

## Exit Code Reference

Exit codes match the `godspeed run` CLI contract documented in `src/godspeed/agent/result.py` (`ExitCode` enum). The `--json-output` flag adds structured metadata (session_id, iterations_used, cost_usd, etc.) to stdout alongside the human-readable response.

## License

MIT — same as the Godspeed coding agent.
