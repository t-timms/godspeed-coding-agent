"""Docker sandbox for isolating agent tool execution.

Provides optional Docker-based containment when the ``docker`` SDK is
installed.  When Docker is unavailable the module degrades to a no-op
so the rest of the codebase never needs to guard imports.

The Docker sandbox is a *technical containment* mechanism — it answers
the question "what can the agent physically reach?".  The approval layer
(PermissionEngine) answers "is this specific tool call allowed right now?".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import docker  # type: ignore[import-untyped]
    import docker.errors

    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False


def is_docker_available() -> bool:
    """Return True if the Docker SDK is importable and a daemon is reachable."""
    if not _DOCKER_AVAILABLE:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Immutable configuration for a Docker sandbox.

    Attributes:
        image: Base image for the container.
        volumes: Host path → container path mount mapping.
        network_mode: Docker network mode (bridge, host, none).
        mem_limit: Memory limit string (e.g. ``"512m"``).
        cpu_period: CPU CFS scheduler period in microseconds.
        cpu_quota: CPU CFS quota in microseconds per period.
        read_only_root: Mount root filesystem read-only.
        working_dir: Working directory inside the container.
        environment: Extra environment variables.
        command: Override command (None = image default).
        remove_on_stop: Remove container when stopped.
        timeout: Default execution timeout in seconds.
        user: UID or username inside the container (default 65534 = nobody).
    """

    image: str = "python:3.12-slim"
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    network_mode: str = "none"
    mem_limit: str = "512m"
    cpu_period: int = 100_000
    cpu_quota: int = 50_000
    read_only_root: bool = False
    working_dir: str = "/workspace"
    environment: dict[str, str] = field(default_factory=dict)
    command: list[str] | None = None
    remove_on_stop: bool = True
    timeout: int = 120
    user: str = "65534"

    def volume_spec(self, container_path: str) -> dict[str, str]:
        """Build a Docker volume mount dict for a single container path."""
        return {
            "bind": container_path,
            "mode": "rw",
        }


class DockerSandbox:
    """Manage a Docker container used as an agent execution sandbox.

    Lifecycle:
        1. ``start()`` — pull image if needed, create and start container.
        2. ``exec_command(cmd)`` — run a command inside the container.
        3. ``stop()`` — stop and optionally remove the container.

    If Docker is unavailable, all methods return safe defaults without
    raising, so callers never need to special-case the no-Docker path.
    """

    def __init__(self, config: DockerSandboxConfig) -> None:
        self._config = config
        self._client: Any = None
        self._container: Any = None

    @property
    def is_running(self) -> bool:
        """True if the container exists and is running."""
        if self._container is None:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False

    @property
    def container_id(self) -> str | None:
        """Short container ID or None."""
        if self._container is None:
            return None
        return self._container.short_id

    def start(self) -> bool:
        """Start the sandbox container. Returns True on success."""
        if not _DOCKER_AVAILABLE:
            logger.warning("Docker SDK not installed — sandbox disabled")
            return False

        try:
            self._client = docker.from_env()
        except Exception:
            logger.warning("Docker daemon unreachable — sandbox disabled")
            return False

        try:
            self._client.images.pull(self._config.image)
        except docker.errors.ImageNotFound:
            logger.error("Docker image not found: %s", self._config.image)
            return False
        except Exception as exc:
            logger.warning("Docker image pull failed: %s", exc)
            return False

        try:
            self._container = self._client.containers.run(
                image=self._config.image,
                command=self._config.command or ["/bin/sleep", "3600"],
                volumes=dict(self._config.volumes) or None,
                network_mode=self._config.network_mode,
                mem_limit=self._config.mem_limit,
                cpu_period=self._config.cpu_period,
                cpu_quota=self._config.cpu_quota,
                read_only=self._config.read_only_root,
                working_dir=self._config.working_dir,
                environment=self._config.environment or None,
                user=self._config.user,
                detach=True,
                remove=self._config.remove_on_stop,
            )
            logger.info(
                "Docker sandbox started container=%s image=%s",
                self._container.short_id,
                self._config.image,
            )
            return True
        except Exception as exc:
            logger.error("Failed to start Docker sandbox: %s", exc)
            return False

    def exec_command(self, command: list[str], timeout: int | None = None) -> tuple[int, str, str]:
        """Execute a command inside the sandbox container.

        Returns:
            (exit_code, stdout, stderr) tuple.
        """
        if not self.is_running:
            return 1, "", "Sandbox container not running"

        effective_timeout = timeout or self._config.timeout
        try:
            result = self._container.exec_run(
                cmd=command,
                demux=True,
                timeout=effective_timeout,
            )
            stdout = (result.output[0] or b"").decode("utf-8", errors="replace")
            stderr = (result.output[1] or b"").decode("utf-8", errors="replace")
            return result.exit_code, stdout, stderr
        except Exception as exc:
            logger.error("Docker exec failed: %s", exc)
            return 1, "", str(exc)

    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if self._container is None:
            return
        try:
            self._container.stop(timeout=5)
        except Exception as exc:
            logger.warning("Docker stop failed: %s", exc)
        try:
            self._container.remove(force=True)
        except Exception as exc:
            logger.warning("Docker remove failed: %s", exc)
        self._container = None
        logger.info("Docker sandbox stopped")
