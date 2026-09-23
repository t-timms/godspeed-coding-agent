"""llama.cpp server management for Godspeed — auto-start, health checks, model discovery.

Mirrors the Ollama manager pattern but for a local llama.cpp OpenAI-compatible
server. The server is expected to expose http://localhost:8080/v1 by default.

Usage from Godspeed CLI:
    _ensure_llamacpp()  # auto-start if not running

Usage from agent (TUI):
    llamacpp status     # check if server is up
    llamacpp start      # start the server
    llamacpp stop       # stop the server
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_API_BASE = f"{DEFAULT_URL}/v1"
LLAMACPP_API_BASE_ENV = "LLAMACPP_API_BASE"

# Default paths — searched in order
DEFAULT_MODELS_DIR = Path.home() / ".llamacpp" / "models"
DEFAULT_SERVER_BINS: list[Path] = [
    # CUDA 12.8 + sm_120a-real + FA build (b9066, Blackwell-optimized)
    Path.home() / ".llamacpp" / "build" / "b9066_cu128" / "llama-server.exe",
    # CUDA 12.8 + sm_120a build (b9066, source rebuild with MMQ fix)
    Path.home() / ".llamacpp" / "build" / "b9066" / "llama-server.exe",
    # Pre-built b9066 from setup script (temporary or alternative location)
    Path.home() / "llama.cpp" / "build" / "b9066" / "llama-server.exe",
    # Locally built (may be older, GPU draft broken in b9046)
    Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server.exe",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
    Path.home() / "llama.cpp" / "build" / "Release" / "llama-server.exe",
]

# Model configuration for auto-start
DEFAULT_MODEL_FILE = "qwen2.5-coder-14b-q4_K_M.gguf"
DEFAULT_CONTEXT = 32768
DEFAULT_GPU_LAYERS = 999

# VRAM-validated ceiling for context when speculative decoding is active,
# specific to the current default model pairing (14B Q4_K main + a 0.5B/1.5B
# draft): 14B Q4_K (~8.1 GiB) + 1.5B Q5_K (~1.2 GiB) + KV cache at 24576
# context (~4.5 GiB) = ~13.8 GiB, fits a 16 GB card with margin. Only a
# *ceiling*, not the value to always use — see start_server's context
# handling below. Re-validate this number if the default model pairing ever
# changes (e.g. Workstream 2's Devstral Small 2 24B + MiniCPM5-2B).
#
# Open question, not resolved here: this math counts KV cache against VRAM,
# but start_server's own no_kv_offload default is True — meaning KV cache
# is placed in system RAM by default, which the original math didn't
# account for. Whether that makes this ceiling overly conservative can only
# be settled by a real run against the actual GPU (no ~/.llamacpp/ build or
# model was available in-session to verify), so the validated 24576 value
# is kept as-is here rather than guessed at.
DRAFT_MODE_MAX_CONTEXT = 24576


def _find_server_binary() -> Path | None:
    """Locate llama-server binary in common paths."""
    for path in DEFAULT_SERVER_BINS:
        if path.exists():
            return path
    # Try PATH
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    return None


def _find_model() -> Path | None:
    """Locate the default GGUF model file."""
    candidate = DEFAULT_MODELS_DIR / DEFAULT_MODEL_FILE
    if candidate.exists():
        return candidate
    # Search models dir for any .gguf
    if DEFAULT_MODELS_DIR.exists():
        ggufs = sorted(DEFAULT_MODELS_DIR.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


def _find_draft_model() -> Path | None:
    """Auto-detect a draft model for speculative decoding.

    Prefers larger draft models (1.5B > 0.5B) because they achieve higher
    acceptance rates (~90% vs ~85%) at the cost of slightly more VRAM.
    Falls back to smaller models or the smallest file overall.
    Returns None if no suitable draft model is found (spec decoding is optional).

    Auto-detection is skipped when the primary model is too large to share
    16 GB VRAM with a draft model (~10 GB threshold).
    """
    if not DEFAULT_MODELS_DIR.exists():
        return None

    # Skip draft for large models (33B+) — insufficient VRAM for both
    main_model = DEFAULT_MODELS_DIR / DEFAULT_MODEL_FILE
    if main_model.exists() and main_model.stat().st_size > 13 * 1024**3:
        return None

    # Priority 1: exact match for Qwen2.5-Coder 1.5B (best acceptance rate)
    candidate = DEFAULT_MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q5_k_m.gguf"
    if candidate.exists():
        return candidate

    # Priority 2: exact match for Qwen2.5-Coder 0.5B (lighter VRAM)
    candidate = DEFAULT_MODELS_DIR / "Qwen2.5-Coder-0.5B-Instruct-Q5_K_M.gguf"
    if candidate.exists():
        return candidate

    # Priority 3: any file with "1.5b" or "1.5B" in the name
    matches = sorted(DEFAULT_MODELS_DIR.glob("*.gguf"))
    for m in matches:
        if "1.5b" in m.name.lower():
            return m

    # Priority 4: any file with "0.5B" or "draft" in the name
    for m in matches:
        if any(p in m.name.lower() for p in ("0.5b", "draft")):
            return m

    # Priority 5: smallest GGUF file (likely a draft model)
    ggufs = sorted(DEFAULT_MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if ggufs and ggufs[0].stat().st_size < 2 * 1024**3:  # < 2GB = likely a draft
        return ggufs[0]

    return None


def is_server_running(url: str = DEFAULT_URL) -> bool:
    """Check if the llama.cpp server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/v1/models", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            return True
    except Exception:
        return False


def start_server(
    *,
    model_path: Path | None = None,
    draft_model_path: Path | None = None,
    context: int = DEFAULT_CONTEXT,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
    flash_attn: bool = True,
    greedy: bool = True,
    port: int = 8080,
    host: str = "127.0.0.1",
    timeout: int = 60,
    no_kv_offload: bool = True,
    kv_cache_type: str = "q8_0",
) -> subprocess.Popen[str] | None:
    """Start llama-server with optimizations. Returns process handle or None.

    Args:
        model_path: Path to the GGUF model. Auto-detected if None.
        draft_model_path: Path to draft model for speculative decoding.
            None = auto-detect draft model file and use (default),
            False = disable,
            Path = explicitly use this GGUF as draft.
        context: Context window size in tokens. Capped at
            DRAFT_MODE_MAX_CONTEXT when a draft model is active (VRAM must
            fit both models plus KV cache) — never raised above what was
            requested, only lowered when it would exceed the validated
            ceiling.
        gpu_layers: Number of layers to offload to GPU (999 = all).
        flash_attn: Enable Flash Attention for faster prompt processing.
        greedy: Use greedy sampling (temp=0, top_k=1) for spec decoding.
        port: Server port (bound to localhost only).
        host: Server bind address (default 127.0.0.1 for security).
        timeout: Seconds to wait for the server to become ready.
        no_kv_offload: Place KV cache in system RAM instead of VRAM.
            Frees ~2-4 GB VRAM for MMQ buffers, improving prompt processing
            by 67-82% on 16 GB cards with large models (33B+).
        kv_cache_type: KV cache quantization type. q8_0 is recommended for
            best quality/speed balance with 96 GB system RAM.
    """
    if is_server_running(f"http://{host}:{port}"):
        logger.info("llama.cpp server already running at %s:%d", host, port)
        return None

    server_bin = _find_server_binary()
    if server_bin is None:
        logger.error(
            "llama-server binary not found. Build llama.cpp first: "
            "python scripts/setup_qwen36_local.py --build-only, "
            "or see ~/.llamacpp/build/ for existing builds"
        )
        return None

    if model_path is None:
        model_path = _find_model()
    if model_path is None:
        logger.error(
            "No GGUF model found. Download Qwen2.5-Coder-14B from "
            "https://huggingface.co/bartowski/qwen2.5-coder-14b-GGUF "
            "and place in %s",
            DEFAULT_MODELS_DIR,
        )
        return None

    # Auto-detect draft model if explicitly requested
    if draft_model_path is None:
        draft_model_path = _find_draft_model()

    # Cap (never raise) context when a draft model is active — see
    # DRAFT_MODE_MAX_CONTEXT's comment for the VRAM math. A single -c flag
    # is emitted below with this value; llama-server takes the *last*
    # occurrence of a repeated flag, so appending a second one later would
    # silently override whatever was decided here regardless of intent.
    effective_context = context
    if draft_model_path and context > DRAFT_MODE_MAX_CONTEXT:
        logger.info(
            "Capping context %d -> %d for speculative decoding (VRAM budget)",
            context,
            DRAFT_MODE_MAX_CONTEXT,
        )
        effective_context = DRAFT_MODE_MAX_CONTEXT

    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "-c",
        str(effective_context),
        "--n-gpu-layers",
        str(gpu_layers),
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Flash Attention — significant speedup on Blackwell GPUs
    if flash_attn:
        cmd.append("--flash-attn")
        cmd.append("on")

    # Greedy sampling — required for high acceptance rate in spec decoding.
    # Only applied when a draft model is active; hurts quality on standalone models.
    if greedy and draft_model_path:
        cmd.append("--sampling-seq")
        cmd.append("k")
        cmd.append("--top-k")
        cmd.append("1")

    # Speculative decoding with draft model
    if draft_model_path:
        cmd.append("-md")
        cmd.append(str(draft_model_path))
        # GPU draft: ~750 tok/s with 1.5B draft on RTX 5070 Ti (b9066+).
        # Requires llama.cpp build b9066 or newer (fixes upstream bug #22337).
        # b9046 and earlier crash with "invalid vector subscript" on GPU draft.
        cmd.append("--n-gpu-layers-draft")
        cmd.append(str(gpu_layers))
        cmd.append("--spec-draft-n-max")
        cmd.append("16")
        cmd.append("--spec-draft-n-min")
        cmd.append("0")
        cmd.append("--draft-p-min")
        cmd.append("0.75")
        # Context is already capped above (effective_context) — do not
        # append a second -c flag here. llama-server takes the *last*
        # occurrence of a repeated flag, so a second -c here would silently
        # override the cap/request decision made above regardless of what
        # it computed. This is exactly the bug that used to force every
        # request down to a hardcoded 24576 even when the caller's context
        # already fit comfortably below it, or explicitly asked for less.

    # Disable auto-fit: user has validated context sizes on this card.
    cmd.append("--fit")
    cmd.append("off")

    # KV cache in system RAM — frees ~2-4 GB VRAM for MMQ kernel buffers.
    # Critical on 16 GB cards running large models: without this, VRAM
    # pressure causes MMQ to stall, dropping throughput significantly.
    # With 96 GB system RAM there is no capacity concern.
    if no_kv_offload:
        cmd.append("--no-kv-offload")

    # KV cache quantization — q8_0 has negligible quality loss vs f16
    # and matches q4_0 speed while providing better attention precision.
    cmd.append("--cache-type-k")
    cmd.append(kv_cache_type)
    cmd.append("--cache-type-v")
    cmd.append(kv_cache_type)

    # Batch size — larger batch gives headroom for long prompts without
    # regressing generation speed on fully-offloaded GPU.
    cmd.append("-b")
    cmd.append("2048")
    cmd.append("-ub")
    cmd.append("512")

    # Thread count — 4 optimal for full GPU offload on Ryzen. More threads
    # add CPU overhead without improving GPU-bound performance.
    cmd.append("-t")
    cmd.append("4")

    # Security: disable web UI, bind to localhost only
    cmd.append("--no-webui")

    logger.info("Starting llama.cpp server: %s", " ".join(cmd))
    try:
        # Performance env vars for Blackwell CUDA 12.8 + MMQ
        env = dict[str, str](os.environ)
        env.setdefault("GGML_CUDA_GRAPH_OPT", "1")
        env.setdefault("GGML_CUDA_FA_ALL_QUANTS", "1")
        env.setdefault("BLACKWELL_NATIVE_FP4", "1")
        # CUDA_SCALE_LAUNCH_QUEUES=4x reduces CPU-side stall overhead by
        # increasing CUDA launch queue capacity on Blackwell GPUs.
        env.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        # CUDA 12.8 conda DLL path for runtime linking (cublas64_12, cudart64_12)
        cuda_bin = Path.home() / "miniconda3" / "envs" / "cuda-build" / "Library" / "bin"
        if cuda_bin.exists():
            env["PATH"] = f"{cuda_bin!s};{env.get('PATH', '')}"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except OSError as exc:
        logger.error("Failed to start llama.cpp server: %s", exc)
        return None

    # Poll until ready
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}"
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if is_server_running(url):
            logger.info("llama.cpp server ready at %s", url)
            return proc
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)
            logger.error("Server exited early. stdout: %s", stdout[:500])
            logger.error("stderr: %s", stderr[:500])
            return None

    logger.warning("llama.cpp server startup timed out after %d seconds", timeout)
    return proc


def stop_server(proc: subprocess.Popen[str] | None) -> bool:
    """Gracefully stop the llama.cpp server process."""
    if proc is None or proc.poll() is not None:
        return True
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Server did not terminate gracefully, killing")
        proc.kill()
        proc.wait(timeout=5)
        return True
    except Exception as exc:
        logger.error("Error stopping server: %s", exc)
        return False


def get_server_status(url: str = DEFAULT_URL) -> dict[str, Any]:
    """Return detailed status of the llama.cpp server."""
    status: dict[str, Any] = {
        "running": False,
        "url": url,
        "model": None,
        "version": None,
    }
    if not is_server_running(url):
        return status

    status["running"] = True

    # Try to read model info from /v1/models
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/v1/models", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read().decode())
            if isinstance(data, dict) and "data" in data:
                models = data["data"]
                if models:
                    status["model"] = models[0].get("id", "unknown")
            elif isinstance(data, list) and data:
                status["model"] = data[0].get("id", "unknown")
    except Exception as exc:
        logger.debug("Could not query /v1/models: %s", exc)

    # Try to read server props from /props
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/props", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read().decode())
            status["version"] = data.get("version", "unknown")
            status["default_generation_settings"] = data.get("default_generation_settings", {})
    except Exception as exc:
        logger.debug("Could not query /props: %s", exc)

    return status


def configure_litellm_env(url: str = DEFAULT_URL) -> None:
    """Set environment variables so LiteLLM routes to the local server.

    This must be called before LiteLLM is imported. Sets:
      - LLAMACPP_API_BASE (used by LiteLLM llamacpp provider)
      - OPENAI_BASE_URL (used by OpenAI v1.0+ / LiteLLM openai/ provider)
    """
    import os

    api_base = f"{url}/v1"
    os.environ.setdefault(LLAMACPP_API_BASE_ENV, api_base)
    os.environ.setdefault("OPENAI_BASE_URL", api_base)
    os.environ.setdefault("OPENAI_API_KEY", "none")
    logger.debug("Configured LiteLLM env: %s=%s", LLAMACPP_API_BASE_ENV, api_base)


class LlamaCppTool(Tool):
    """Manage the local llama.cpp inference server.

    Provides status checks, start, and stop actions for the OpenAI-compatible
    server that powers local model inference.
    """

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def description(self) -> str:
        return (
            "Manage the local llama.cpp inference server. "
            "Actions: status (check server health), start (launch server), "
            "stop (shutdown server). Use this to verify local model availability."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "start", "stop"],
                    "description": "Action: status, start, or stop.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "status")

        if action == "status":
            status = get_server_status()
            if status["running"]:
                lines = ["llama.cpp server: RUNNING"]
                lines.append(f"  URL: {status['url']}")
                if status["model"]:
                    lines.append(f"  Model: {status['model']}")
                if status["version"]:
                    lines.append(f"  Version: {status['version']}")
                return ToolResult.ok("\n".join(lines))
            return ToolResult.ok(
                f"llama.cpp server: NOT RUNNING\n  URL: {status['url']}\n"
                "  Run 'llamacpp start' to launch."
            )

        if action == "start":
            proc = start_server()
            if proc is not None:
                return ToolResult.ok("llama.cpp server started successfully.")
            if is_server_running():
                return ToolResult.ok("llama.cpp server was already running.")
            return ToolResult.failure(
                "Failed to start llama.cpp server. "
                "Ensure the model is downloaded and llama.cpp is built."
            )

        if action == "stop":
            # We don't have the process handle in the tool context,
            # so we can't stop a server we didn't start. This is a
            # limitation; the CLI manages the lifecycle.
            return ToolResult.ok(
                "llama.cpp server stop requested. "
                "(If started by Godspeed CLI, it will stop on exit.)"
            )

        return ToolResult.failure(f"Unknown action: {action!r}")
