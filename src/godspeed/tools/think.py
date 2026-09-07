"""Think tool — a no-op scratchpad for the model's reasoning.

Lets the model write down its strategy, parse complex tool output, or plan
before acting. It changes nothing in the environment and returns a short
confirmation instead of echoing the thought back, keeping context cost near
zero (the model already holds the thought in its own context).
"""

from __future__ import annotations

from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

#: Maximum length of a single thought. Longer thoughts are rejected so the
#: model must condense its reasoning rather than silently losing the tail.
MAX_THOUGHT_CHARS = 8000

#: Short confirmation returned on success. Deliberately does NOT echo the
#: thought text back — the model already has it, so echoing would only burn
#: context tokens for no benefit.
_CONFIRMATION = "Thought recorded. Continue with your plan."


class ThinkTool(Tool):
    """Record a thought / plan without changing the environment.

    Pure reasoning scratchpad: the model writes down strategy, parses complex
    tool output, or plans next steps. Read-only and side-effect free, so it is
    auto-approved in every non-plan-blocked mode.
    """

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return (
            "Write down your strategy, parse complex tool output, or plan "
            "your next actions before acting. This tool changes nothing in "
            "the environment — it only records a thought for your own "
            "reasoning. Use it to structure multi-step work or to summarize "
            "what a large tool result means before proceeding.\n\n"
            "Example: think(thought='The build failed on a missing import; "
            "next I will check the traceback and fix the import path.')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "The thought, strategy, or plan to record. Required. "
                        f"Must be a non-empty string up to {MAX_THOUGHT_CHARS} "
                        "characters."
                    ),
                },
                "next_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional ordered list of next actions to take. "
                        "Ignored by the tool; purely for the model's own "
                        "planning."
                    ),
                },
            },
            "required": ["thought"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        thought = arguments.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            return ToolResult.failure("thought must be a non-empty string")

        if len(thought) > MAX_THOUGHT_CHARS:
            return ToolResult.failure(
                f"thought exceeds maximum length of {MAX_THOUGHT_CHARS} characters "
                f"(got {len(thought)}). Condense your thought and retry."
            )

        # Everything else in arguments (e.g. next_actions) is intentionally
        # ignored — the tool only records the thought and changes nothing.
        return ToolResult.ok(_CONFIRMATION)
