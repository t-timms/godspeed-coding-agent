"""Sandbox package — technical containment separate from approval logic.

Sandbox defines *what the agent can technically do* (writable paths,
network access, Docker container settings). Approval (via the existing
PermissionEngine) defines *what the agent is allowed to do now*.
The ``SandboxPolicy`` in ``policy.py`` bridges both dimensions.
"""
