"""Shared helpers for diff-based review commands (/code-review, /security-review, /simplify).

Pure, unit-testable seams: diff collection accepts an injected git runner so
tests never invoke subprocess; secrets scanning runs the deterministic detector
from ``godspeed.security.secrets`` and works without any LLM.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from godspeed.security.secrets import detect_secrets

logger = logging.getLogger(__name__)

MAX_DIFF_LINES = 400
MAX_SCAN_FILES = 20
GIT_TIMEOUT = 10

MODE_PROMPTS: dict[str, str] = {
    "code": (
        "You are a strict code reviewer. Review the following diff and list "
        "findings: bugs, security risks, performance issues, and best-practice "
        "violations. For each finding give file:line, severity "
        "(high/medium/low), and a one-sentence fix suggestion. If there are "
        "no issues, respond exactly: NO ISSUES"
    ),
    "security": (
        "You are a security reviewer. Review the following diff and list "
        "security vulnerabilities only: injection, secret leakage, auth "
        "bypass, path traversal, unsafe subprocess or deserialization. For "
        "each finding give file:line, severity (high/medium/low), and a "
        "one-sentence remediation. If there are none, respond exactly: "
        "NO ISSUES"
    ),
    "simplify": (
        "You are a cleanup reviewer. Review the following diff and list only "
        "cleanup opportunities: dead code, duplication, poor naming, "
        "needless complexity. Do not hunt for bugs. For each finding give "
        "file:line plus a one-sentence suggestion. If there are none, "
        "respond exactly: NO ISSUES"
    ),
}

GitRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class ReviewDiff:
    """A collected diff plus parse metadata."""

    stat_summary: str = ""
    diff_text: str = ""
    truncated: bool = False
    changed_files: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _default_runner(args: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def collect_diff(cwd: Path, runner: GitRunner | None = None) -> ReviewDiff:
    """Collect the working-tree diff via git; ``runner`` is the test seam."""
    run = runner or _default_runner

    try:
        repo_check = run(["git", "rev-parse", "--git-dir"], cwd=cwd, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return ReviewDiff(error=f"git unavailable: {exc}")
    if repo_check.returncode != 0:
        return ReviewDiff(error="Not a git repository.")

    try:
        status = run(["git", "status", "--porcelain"], cwd=cwd, timeout=GIT_TIMEOUT)
        diff = run(["git", "diff", "HEAD"], cwd=cwd, timeout=GIT_TIMEOUT)
        stat = run(["git", "diff", "--stat", "HEAD"], cwd=cwd, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return ReviewDiff(error=f"git command failed: {exc}")

    if status.returncode != 0:
        return ReviewDiff(error=f"git status failed: {status.stderr.strip()}")

    changed_files = [line[3:] for line in status.stdout.splitlines() if len(line) > 3]
    raw_diff = diff.stdout if diff.returncode == 0 else ""
    lines = raw_diff.splitlines()
    truncated = len(lines) > MAX_DIFF_LINES

    return ReviewDiff(
        stat_summary=stat.stdout.strip() if stat.returncode == 0 else "",
        diff_text="\n".join(lines[:MAX_DIFF_LINES]),
        truncated=truncated,
        changed_files=changed_files,
    )


def review_prompt(mode: str, diff: ReviewDiff, extra: str = "") -> str:
    """Build the single-shot review prompt for ``mode``.

    Raises ValueError on unknown modes or unusable diffs.
    """
    if mode not in MODE_PROMPTS:
        raise ValueError(f"Unknown review mode: {mode}")
    if diff.error:
        raise ValueError(f"Cannot review: {diff.error}")

    prompt = MODE_PROMPTS[mode]
    if not diff.diff_text.strip():
        return f"{prompt}\n{extra}".rstrip() if extra else prompt

    diff_note = f"\n\n(Diff truncated to {MAX_DIFF_LINES} lines.)" if diff.truncated else ""
    stat_note = f"\n\nChanged files:\n{diff.stat_summary}" if diff.stat_summary else ""
    extra_note = f"\n\nAdditional instructions: {extra}" if extra else ""
    return f"{prompt}{stat_note}{diff_note}\n\nThe diff:\n{diff.diff_text}{extra_note}"


def security_scan_files(
    files: list[str],
    cwd: Path,
    reader: Callable[[Path], str] | None = None,
) -> list[str]:
    """Run deterministic secret detection over changed file paths.

    Files beyond the cap, missing, or unreadable are skipped quietly
    (debug log); scanning never raises.
    """
    resolved: list[str] = []
    for rel_path in files[:MAX_SCAN_FILES]:
        path = Path(rel_path)
        full = path if path.is_absolute() else cwd / path
        try:
            if reader is not None:
                content = reader(full)
            else:
                content = full.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("Secret scan skipped file=%s error=%s", full, exc)
            continue

        for finding in detect_secrets(content):
            snippet = finding.match[:40].replace("\n", " ")
            resolved.append(f"[secrets] {rel_path}: {finding.secret_type}: {snippet}")
    return resolved


def parse_review_args(args: str) -> tuple[list[str], str]:
    """Split ``--flags`` from positional text in review command args."""
    tokens = args.split()
    flags = [tok for tok in tokens if tok.startswith("--")]
    positional = " ".join(tok for tok in tokens if not tok.startswith("--"))
    return flags, positional


def findings_from_response(content: str) -> list[str]:
    """Convert a single-shot review response into finding lines.

    ``NO ISSUES`` (case-insensitive) maps to an empty list; other content
    becomes one bullet per non-empty line with markdown list markers
    stripped, falling back to a single-entry list for prose answers.
    """
    text = content.strip()
    if not text:
        return []
    if text.upper().startswith("NO ISSUES"):
        return []
    bullets = [
        line.strip().lstrip("-*").strip()
        for line in text.splitlines()
        if line.strip().lstrip("-*").strip()
    ]
    return bullets if bullets else [text]


def format_findings(findings: list[str], empty_note: str) -> str:
    """Format finding lines for panel display; used by review commands."""
    if not findings:
        return empty_note
    return "\n".join(f"- {f}" for f in findings)
