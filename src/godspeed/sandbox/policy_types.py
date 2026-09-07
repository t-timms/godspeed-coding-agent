"""Sandbox policy data types.

``NetworkRule`` and ``SandboxPolicy`` are pure dataclasses with no
dependency on the tool layer.  They live in their own leaf module so that
``godspeed.tools.base`` can import ``SandboxPolicy`` at runtime (needed by
Pydantic to build ``ToolContext``) without creating a circular import —
``godspeed.sandbox.policy`` imports ``ToolCall`` from the tool layer.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from godspeed.sandbox.docker import DockerSandboxConfig


@dataclass
class NetworkRule:
    """A single network access rule.

    Attributes:
        pattern: Glob pattern for hostnames (``*.github.com``, ``localhost``).
        ports: Allowed ports (empty = all ports).
        action: ``"allow"`` or ``"deny"``.
    """

    pattern: str
    ports: list[int] = field(default_factory=list)
    action: str = "allow"

    def matches(self, hostname: str, port: int | None = None) -> bool:
        """Check if this rule matches a hostname:port combination."""
        if not fnmatch.fnmatch(hostname, self.pattern):
            return False
        return not (self.ports and port is not None and port not in self.ports)


@dataclass
class SandboxPolicy:
    """Technical containment configuration.

    Defines the boundaries of what the agent can physically access,
    independent of per-call approval decisions.

    Attributes:
        writable_paths: Directories the agent may write to.
        readable_paths: Directories the agent may read (empty = all).
        blocked_paths: Paths the agent can never access (deny-first).
        network_rules: Ordered network access rules (first match wins).
        docker: Optional Docker sandbox configuration.
        enable_network: Global network toggle.
    """

    writable_paths: list[str] = field(default_factory=list)
    readable_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=list)
    network_rules: list[NetworkRule] = field(default_factory=list)
    docker: DockerSandboxConfig | None = None
    enable_network: bool = True

    def is_path_writable(self, path: str) -> bool:
        """Check if a path falls within any writable directory."""
        normalized = str(Path(path).resolve())
        if self._is_blocked(normalized):
            return False
        if not self.writable_paths:
            return True
        norm = self._norm(normalized)
        return any(
            norm.startswith(self._norm(str(Path(wp).resolve()))) for wp in self.writable_paths
        )

    def is_path_readable(self, path: str) -> bool:
        """Check if a path is readable (empty readable_paths = all allowed)."""
        normalized = str(Path(path).resolve())
        if self._is_blocked(normalized):
            return False
        if not self.readable_paths:
            return True
        norm = self._norm(normalized)
        return any(
            norm.startswith(self._norm(str(Path(rp).resolve()))) for rp in self.readable_paths
        )

    @staticmethod
    def _norm(s: str) -> str:
        return s.casefold() if os.name == "nt" else s

    def _is_blocked(self, normalized: str) -> bool:
        """Check if a normalized path matches any blocked entry.

        Entries may be absolute/relative paths (separator-aware prefix
        match), globs (fnmatch against full path or basename), or bare
        filenames (basename equality at any depth). Comparisons are
        case-insensitive on Windows.
        """
        norm = self._norm(normalized)
        name_norm = self._norm(Path(normalized).name)
        for bp in self.blocked_paths:
            has_sep = "/" in bp or "\\" in bp or ":" in bp or bp.startswith("~")
            if has_sep:
                resolved = str(Path(bp).expanduser().resolve())
                rn = self._norm(resolved)
                if norm == rn or norm.startswith(rn + os.sep) or norm.startswith(rn + "/"):
                    return True
            if any(ch in bp for ch in "*?["):
                pat = self._norm(bp)
                if fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(name_norm, pat):
                    return True
                continue
            if not has_sep and name_norm == self._norm(bp):
                return True
        return False

    def is_network_allowed(self, hostname: str, port: int | None = None) -> bool:
        """Check if network access is allowed for a given host:port.

        Rules are evaluated in order; first match wins.
        If no rules match, falls back to ``enable_network``.
        """
        if not self.enable_network:
            return False
        for rule in self.network_rules:
            if rule.matches(hostname, port):
                return rule.action == "allow"
        return self.enable_network
