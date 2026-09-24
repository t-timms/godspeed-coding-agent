# Benchmark hygiene: isolating the agent

A benchmark agent has a shell. Two things go wrong when that shell is the host's, both seen in a
real SWE-bench Lite evaluation of local models:

* **The agent installs into your harness.** It runs `pip install -e .` and `pip install "setuptools<81"`.
  With Godspeed's own virtualenv first on `PATH`, those installs land in Godspeed's venv and leak from one
  task into the next.
* **The agent goes looking for the answer.** After the verify tool printed a failing test name, one model
  ran `find / -name "test_datetime_list_inner_format"`, `grep -rn <that name> /home/...`, greps of the
  `swebench` package for `FAIL_TO_PASS`, `find / -name "gold_results*"` and `find / -name "*1359*"` (the
  instance number). Gold patches, the dataset cache and other runs' predictions live on that filesystem.
  In that unsandboxed run 21 of its 376 shell commands left the workspace, versus 1 of 165 for another
  model under the same harness and shell.

## Flags (`experiments/swebench_lite/run.py`)

```bash
python experiments/swebench_lite/run.py --agent-in-loop --model openai/qwen3.8-27b \
    --split dev --isolate-agent-shell --task-python 3.9
```

* `--task-python VERSION` gives every task its own throwaway virtualenv (created with `uv`, `pytest`
  preinstalled, first on `PATH` for the agent's commands, deleted afterwards). Use the Python version of the
  SWE-bench images (3.9). Godspeed itself keeps running from its own interpreter.
* `--isolate-agent-shell` sets `GODSPEED_SHELL_WRAPPER=scripts/agent_shell_isolate.sh`, so **every agent
  command** runs in a private mount + PID namespace where:
  * `$HOME` and `/mnt` are empty tmpfs (only the uv-managed Python interpreters are visible, read-only),
  * `/tmp` is a per-task private directory that persists across that task's commands,
  * the task workspace and its venv are visible and writable, and edits reach the host (patch capture),
  * `/etc/resolv.conf` is kept (on WSL2 it points into `/mnt/wsl`; without it the agent silently loses DNS),
  * all capabilities are dropped, so the agent cannot unmount anything.

`GODSPEED_SHELL_WRAPPER` is a general hook in the shell tool (POSIX only): the wrapper is called as
`WRAPPER -c COMMAND`. Environment knobs of the script (`AGENT_PRIV`, `AGENT_VENV`, `AGENT_KEEP_RO`,
`AGENT_MASK`, `AGENT_REAL_HOME`) are documented at the top of `scripts/agent_shell_isolate.sh`.

Requirements: Linux, util-linux `unshare` and `setpriv`, unprivileged user namespaces (on Ubuntu 24.04
check `kernel.apparmor_restrict_unprivileged_userns`), and `uv` for `--task-python`. The runner checks
once at startup and **fails closed**: if the sandbox cannot be created, commands are not run unsandboxed.

## Audit every arm

```bash
python scripts/audit_agent_commands.py run.log --allow /path/to/task/venvs --term my-gold-dir
```

Flags recursive searches over `/`, `~` or host trees (`wide-search`), absolute paths under `/home`,
`/mnt`, `/root`, ... (`outside-path`), and mentions of `FAIL_TO_PASS`, `test_patch`, harness internals or
dataset caches (`hunt-term`). Treat a task with hits as unverified unless the shell was isolated (then hits
are informational: the search found nothing). `--fail-on-flag` exits 1 for use in scripts.

## What this does not do

* It is not a security boundary against a determined adversary: the kernel, CPU and **network are shared**.
  An agent with network access can still fetch upstream history or the fix from GitHub; restrict egress
  separately if that matters for your claims.
* The in-loop verify tool prints failing test names and output by design, so scores from
  `--agent-in-loop` are not comparable to leaderboards that give the agent no oracle.
* Score patches with the official harness afterwards rather than trusting the in-loop verdict.
