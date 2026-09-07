"""Deep analysis tool: 3-step structured reasoning (generate -> critique -> refine).

Uses the agent's existing LLM client for consistency — no separate API key
or raw HTTP calls.
"""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class DeepAnalysisTool(Tool):
    """Structured reasoning for complex coding problems.

    Three-step process using the active LLM:
    1. Generate initial solution from problem + context
    2. Critique the solution for weaknesses, bugs, edge cases
    3. Refine to produce an improved solution addressing the critique

    Returns the refined solution with all intermediate reasoning visible
    for the calling agent to use.
    """

    def __init__(self, llm_client: Any | None = None):
        self._llm_client = llm_client

    @property
    def name(self) -> str:
        return "deep_analysis"

    @property
    def description(self) -> str:
        return (
            "Perform structured 3-step reasoning on a coding problem: "
            "(1) generate initial solution, (2) critique it for weaknesses "
            "and edge cases, (3) refine into an improved solution. "
            "Use before making complex code changes to improve first-attempt "
            "correctness. Arguments: problem_statement (required, str), "
            "context (optional, str — file contents, error messages, etc.)."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "problem_statement": {
                    "type": "string",
                    "description": "The problem description or issue to solve",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context: file contents, error messages, test output",
                },
            },
            "required": ["problem_statement"],
            "examples": [
                {
                    "problem_statement": (
                        "Fix IndexError in data_loader.py when input list is empty"
                    ),
                    "context": (
                        "File data_loader.py line 42: items = data[0]  # crashes on empty input"
                    ),
                }
            ],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        problem = arguments.get("problem_statement", "")
        ctx = arguments.get("context", "")

        if not problem:
            return ToolResult.failure("problem_statement is required")

        llm = self._llm_client or getattr(context, "llm_client", None)
        if llm is None:
            return ToolResult.failure(
                "deep_analysis requires an LLM client. "
                "Pass one via the constructor or set it on the ToolContext."
            )

        try:
            step1_prompt = _build_step1_prompt(problem, ctx)
            step2_prompt, initial_solution = await _run_step(llm, step1_prompt, "Step 1: Generate")
            if isinstance(initial_solution, ToolResult):
                return initial_solution

            step2_prompt = _build_step2_prompt(problem, initial_solution)
            _, critique = await _run_step(llm, step2_prompt, "Step 2: Critique")
            if isinstance(critique, ToolResult):
                pass  # critique already has the text in error attribute

            step3_prompt = _build_step3_prompt(problem, initial_solution, str(critique))
            _, refined = await _run_step(llm, step3_prompt, "Step 3: Refine")
            refined_text = initial_solution if isinstance(refined, ToolResult) else str(refined)

            result = (
                f"## Initial Solution\n{initial_solution}\n\n"
                f"## Critique\n{critique}\n\n"
                f"## Refined Solution (use this)\n{refined_text}"
            )
            return ToolResult.success(result)

        except Exception as exc:
            logger.exception("deep_analysis failed")
            return ToolResult.failure(f"deep_analysis failed: {exc}")


async def _run_step(llm: Any, prompt: str, step_label: str) -> tuple[str, str | ToolResult]:
    """Run one step of the analysis pipeline. Returns (prompt_text, response_content).

    On success the second element is the response string. On failure it is a
    ToolResult so the caller can propagate the error.
    """
    logger.info("[deep_analysis] %s", step_label)
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await llm.chat(messages=messages, max_tokens=4096)
        return prompt, response.content
    except Exception as exc:
        logger.error("[deep_analysis] %s failed: %s", step_label, exc)
        return prompt, ToolResult.failure(f"{step_label} failed: {exc}")


def _build_step1_prompt(problem: str, context: str) -> str:
    return (
        "You are an expert software engineer. Given the problem below, "
        "provide a detailed, minimal solution.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Context:\n{context}\n\n"
        "Provide a complete, minimal solution that fixes the issue. "
        "Include specific file paths and the exact code changes needed. "
        "Keep the fix as small as possible — do not refactor unrelated code."
    )


def _build_step2_prompt(problem: str, solution: str) -> str:
    return (
        "Review the following solution and identify ALL weaknesses, bugs, and limitations.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n{solution}\n\n"
        "Critique requirements:\n"
        "- Identify any bugs or incorrect logic\n"
        "- Note edge cases not handled (None, empty, nested, unicode, etc.)\n"
        "- Check if the fix addresses the ROOT CAUSE, not just symptoms\n"
        "- Note any changes that might break working functionality\n"
        "- Be specific and thorough."
    )


def _build_step3_prompt(problem: str, solution: str, critique: str) -> str:
    return (
        "Improve the following solution based on the critique.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Original Solution:\n{solution}\n\n"
        f"Critique:\n{critique}\n\n"
        "Provide a refined solution that addresses ALL weaknesses identified. "
        "The solution should be minimal, robust, and effective. "
        "Include the exact file paths and code changes."
    )
