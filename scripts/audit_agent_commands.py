#!/usr/bin/env python3
"""Audit a benchmark run log for agent shell commands that look for the answer outside the task.

Godspeed logs every agent command as ``shell.execute command='...' timeout=N``. Agents have been
seen running ``find / -name "*1359*"`` (an instance number), ``grep -rn <failing test name>
/home/...`` and greps of the swebench package for ``FAIL_TO_PASS``: looking for hidden tests or
gold patches on the host. Run this after every arm. Treat a task with flagged commands as
unverified unless the agent shell was isolated (``scripts/agent_shell_isolate.sh``), in which case
the search could not have found anything and hits are informational.

Usage:
    audit_agent_commands.py LOG [LOG ...] [--allow PREFIX ...] [--term TEXT ...] [--show N]
                            [--fail-on-flag]

A command is flagged for one or more reasons:
    wide-search   recursive find/grep/rg over / or ~ or a host tree (/home, /mnt, ...)
    outside-path  an absolute path under /home, /mnt, /root, /var, /opt or /srv that is not allowed
    hunt-term     mentions FAIL_TO_PASS, PASS_TO_PASS, test_patch, gold results, swebench
                  internals or dataset caches
Paths under /tmp are workspace/venv territory and never flagged. Add ``--allow`` prefixes for
anything else that is legitimate (for example the run's own venv directory) and ``--term`` for
project-specific strings that must never appear (your run or gold-data directory names).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field

COMMAND_RE = re.compile(r"shell\.execute command=(?P<cmd>.+?) timeout=\d+")
# A recursive search whose target is / or ~ or a host tree agents have no business in.
_WIDE_TARGET = r"(?:/|~|/(?:home|mnt|root|var|opt|srv|Users)(?:/[^\s'\"]*)?)"
WIDE_SEARCH_RE = re.compile(
    rf"\b(?:find|grep\s+-\w*[rR]\w*|rg|fd|locate)\b[^|;&]*?\s{_WIDE_TARGET}(?=[\s'\"]|$)"
)
ABS_PATH_RE = re.compile(
    r"(?<![\w./-])(/(?:home|mnt|root|var|opt|srv|Users)(?:/[^\s'\"`;|&)<>]*)?)"
)
HUNT_TERMS = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "test_patch",
    "gold_results",
    "gold patch",
    "swebench/harness",
    "site-packages/swebench",
    "huggingface",
    "predictions_",
)


@dataclass
class Flagged:
    line_no: int
    command: str
    reasons: list[str] = field(default_factory=list)


def classify(command: str, allow: Iterable[str] = (), extra_terms: Iterable[str] = ()) -> list[str]:
    """Return the reasons this command looks like a search outside the task (empty = fine)."""
    reasons: list[str] = []
    if WIDE_SEARCH_RE.search(command):
        reasons.append("wide-search")
    allowed = tuple(allow)
    for path in ABS_PATH_RE.findall(command):
        if not (allowed and path.startswith(allowed)):
            reasons.append("outside-path")
            break
    lowered = command.lower()
    if any(term.lower() in lowered for term in (*HUNT_TERMS, *extra_terms)):
        reasons.append("hunt-term")
    return reasons


def audit(
    lines: Iterable[str], allow: Iterable[str] = (), extra_terms: Iterable[str] = ()
) -> tuple[int, list[Flagged]]:
    """Return (total commands, flagged commands) for the given log lines."""
    total = 0
    flagged: list[Flagged] = []
    allow_t = tuple(allow)
    terms = tuple(extra_terms)
    for i, line in enumerate(lines, 1):
        m = COMMAND_RE.search(line)
        if not m:
            continue
        total += 1
        cmd = m.group("cmd")
        reasons = classify(cmd, allow_t, terms)
        if reasons:
            flagged.append(Flagged(i, cmd, reasons))
    return total, flagged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+")
    ap.add_argument(
        "--allow", action="append", default=[], help="path prefix that is fine (repeatable)"
    )
    ap.add_argument(
        "--show", type=int, default=10, help="how many flagged commands to print per log"
    )
    ap.add_argument(
        "--term",
        action="append",
        default=[],
        help="extra hunt term to flag, e.g. the name of your run directory (repeatable)",
    )
    ap.add_argument("--fail-on-flag", action="store_true", help="exit 1 if anything was flagged")
    args = ap.parse_args(argv)

    any_flagged = False
    for path in args.logs:
        with open(path, encoding="utf-8", errors="replace") as fh:
            total, flagged = audit(fh, args.allow, args.term)
        print(f"{path}: {len(flagged)} flagged / {total} shell commands")
        for f in flagged[: args.show]:
            print(f"  line {f.line_no} [{','.join(f.reasons)}] {f.command[:160]}")
        any_flagged = any_flagged or bool(flagged)
    return 1 if (any_flagged and args.fail_on_flag) else 0


if __name__ == "__main__":
    sys.exit(main())
