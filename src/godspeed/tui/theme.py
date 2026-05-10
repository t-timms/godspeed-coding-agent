"""Godspeed visual identity — Earth-tone palette.

Design philosophy:
- Earth tones: warm browns, sage greens, terracotta, ochre
- Natural, grounded aesthetic — less clinical than cool blues
- Semantic colors drawn from nature (clay, leaf, amber, bark)
"""

from __future__ import annotations

# =============================================================================
# Core palette — earth tones
# =============================================================================

# Primary accent — warm terracotta clay
# Used for: brand name, active tool names, command names, headers,
#           current model, important values, plan mode.
PRIMARY = "#c17c5b"

# Neutral — warm brown-grey
# Used for: structural labels (Model, Project, Mode), borders,
#           separators, metadata, inactive elements.
NEUTRAL = "#8b7355"

# Semantic states — natural earth tones
SUCCESS = "#87a96b"  # Sage green — ok, enabled, normal
ERROR = "#a45a52"  # Terracotta red — errors, denied, critical
WARNING = "#d4a817"  # Ochre amber — warnings, strict, ask

# Terminal-native dim for content text
DIM = "dim"

# =============================================================================
# ANSI equivalents — prompt-toolkit HTML (hex unsupported there)
# =============================================================================

ANSI_PRIMARY = "ansibrown"
ANSI_NEUTRAL = "ansibrightblack"
ANSI_SUCCESS = "ansigreen"
ANSI_ERROR = "ansired"
ANSI_WARNING = "ansiyellow"
ANSI_BRIGHT_PRIMARY = "ansiyellow"

# =============================================================================
# Semantic styles — bold variants for emphasis
# =============================================================================

BOLD_PRIMARY = f"bold {PRIMARY}"
BOLD_SUCCESS = f"bold {SUCCESS}"
BOLD_ERROR = f"bold {ERROR}"
BOLD_WARNING = f"bold {WARNING}"

# =============================================================================
# Panel borders — unified
# =============================================================================

BORDER_BRAND = PRIMARY
BORDER_TOOL = NEUTRAL
BORDER_INFO = NEUTRAL
BORDER_SUCCESS = SUCCESS
BORDER_ERROR = ERROR
BORDER_WARNING = WARNING

# =============================================================================
# Table styles
# =============================================================================

TABLE_HEADER = f"bold {NEUTRAL}"
TABLE_BORDER = NEUTRAL
TABLE_KEY = NEUTRAL
TABLE_VALUE = "bold"

# =============================================================================
# Permission colors — semantic mapping
# =============================================================================

PERM_ALLOW = SUCCESS
PERM_DENY = ERROR
PERM_ASK = WARNING
PERM_SESSION = PRIMARY

# =============================================================================
# Context-window usage colors — traffic-light with earth tones
# =============================================================================

CTX_OK = SUCCESS
CTX_WARN = WARNING
CTX_CRITICAL = ERROR

# =============================================================================
# Branded strings
# =============================================================================

PROMPT_ICON = ">"
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = "Build fast"

# Syntax theme
SYNTAX_THEME = "monokai"

# =============================================================================
# Ultra-clean markers — Text-only, no emoji
# =============================================================================

MARKER_SUCCESS = "ok"
MARKER_ERROR = "x"
MARKER_WARNING = "!"
MARKER_TOOL = ">"
MARKER_INFO = "i"
MARKER_PARALLEL = "||"
SEPARATOR_DOT = "|"

# Structural
DECORATOR = ""
RULE_CHAR = "-"
GUTTER = ""
GUTTER_STYLE = NEUTRAL

# =============================================================================
# Markup helpers
# =============================================================================


def styled(text: str, style: str) -> str:
    """Wrap text in Rich markup tags."""
    return f"[{style}]{text}[/{style}]"


def brand(version: str = "") -> str:
    """Return the branded product name with optional version."""
    name = styled("Godspeed", BOLD_PRIMARY)
    if version:
        return f"{name} {styled(f'v{version}', NEUTRAL)}"
    return name


def icon_prompt(
    state: str = "",
    turn: int = 0,
    context_pct: float = 0.0,
    compact: bool = False,
    model: str = "",
    cost: float = 0.0,
) -> str:
    """Return the branded prompt string for prompt-toolkit (HTML format).

    State can be: '' (normal), 'plan' (plan mode), 'paused'.
    When *turn* > 0 and *compact* is False, the prompt includes the turn
    number, context-window usage percentage, model, and cost.
    """
    color = ANSI_PRIMARY
    icon_c = PROMPT_ICON
    suffix = ""
    if state == "plan":
        suffix = " [plan]"
        color = ANSI_BRIGHT_PRIMARY
    elif state == "paused":
        suffix = " [paused]"
        color = ANSI_WARNING

    prompt = f"<b><{color}>{icon_c} {PROMPT_TEXT}{suffix}></{color}></b>"

    if not compact and turn > 0:
        extras: list[str] = []

        # Model short name
        if model:
            short = model.split("/", 1)[-1] if "/" in model else model
            extras.append(f"<span color=\"{ANSI_NEUTRAL}\">{short}</span>")

        # Cost
        if cost > 0:
            cost_str = f"${cost:.4f}" if cost >= 0.01 else f"${cost:.6f}"
            extras.append(f"<span color=\"{ANSI_NEUTRAL}\">{cost_str}</span>")

        # Turn + context
        extras.append(f"turn {turn}")
        if context_pct > 0:
            if context_pct >= 90:
                ctx_color = ANSI_ERROR
            elif context_pct >= 70:
                ctx_color = ANSI_WARNING
            else:
                ctx_color = ANSI_SUCCESS
            extras.append(f"ctx <{ctx_color}>{context_pct:.0f}%</{ctx_color}>")
        prompt = f'<span color="{ANSI_NEUTRAL}">{" | ".join(extras)}</span> {prompt}'

    return prompt + " "
