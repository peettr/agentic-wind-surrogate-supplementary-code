"""Adapters that turn a scout prompt into a structured proposal list.

Every caller has the same signature::

    def call_X(prompt: str, *, timeout: int = 600) -> dict[str, Any]

and returns a dict with a ``"proposals"`` key (may be empty on failure).
We support three transport families:

* **CLI-based callers** (``call_claude``, ``call_codex``) -- spawn the local
  ``claude`` / ``codex`` binary via ``subprocess``, pipe the prompt on stdin,
  collect stdout, and extract the JSON object. These tools already have
  built-in web browsing.

* **Gemini CLI callers** (``call_gemini``, ``call_deep_research``) -- use the
  local ``gemini`` binary with Google OAuth login (``GOOGLE_GENAI_USE_GCA=true``).
  ``call_gemini`` uses built-in Google Search grounding; ``call_deep_research``
  invokes the ``gemini-deep-research`` extension (requires paid API key +
  monkey-patched loop detection). Both require ``--yolo`` for automatic tool
  approval.

* **API-based callers** (``call_glm``, ``call_deepseek``, ``call_mimo``,
  ``call_grok``) -- read an API key from OpenClaw ``models.json`` and POST to
  the vendor endpoint. When no key is configured the caller returns an empty
  proposal list with a ``"skipped"`` status so scouting continues.

No caller raises on network or parse errors -- the whole point of running
many scouts in parallel is to tolerate the flaky ones.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib import request as _urlrequest
from urllib import error as _urlerror


LOGGER = logging.getLogger("model_scout.ai_callers")

DEFAULT_TIMEOUT_S = 600


# =======================================================================
# Response parsing helpers
# =======================================================================
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull JSON proposals out of a model response.

    Strategies (first match wins):
      1. A fenced ```json ... ``` block.
      2. All balanced-brace substrings -- prefer the one with ``proposals``.
      3. Bare JSON array.
      4. Prose fallback (Deep Research style).
    """
    if not text:
        return {"proposals": []}
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            parsed = json.loads(stripped)
            out = _coerce_to_proposal_dict(parsed)
            if out.get("proposals"):
                return out
        except json.JSONDecodeError:
            pass
    # 1. Fenced block.
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return _coerce_to_proposal_dict(parsed)
        except json.JSONDecodeError:
            pass
    # 2. Collect ALL balanced-brace substrings, pick the best one.
    # BUG FIX: old code used last { first, which found the innermost
    # object in an array (e.g. {"name":"C"} inside [{A},{B},{C}]),
    # losing all but one proposal.
    candidates: list[tuple[int, dict]] = []
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            candidates.append((start, parsed))
                    except json.JSONDecodeError:
                        pass
                    break
    if candidates:
        # Prefer objects that have a non-empty "proposals" list
        for _, obj in candidates:
            pl = obj.get("proposals")
            if isinstance(pl, list) and len(pl) > 0:
                return _coerce_to_proposal_dict(obj)
        # Do not fall back to an inner proposal object here. If the response is
        # a top-level JSON array, that array parser below should get a chance.
    # 3. Bare JSON array.
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        return _coerce_to_proposal_dict(parsed)
                    except json.JSONDecodeError:
                        break
    # 4. Prose fallback -- extract entries from long-form text.
    proposals = _extract_from_prose(text)
    if proposals:
        return {"proposals": proposals, "status": "ok"}

    LOGGER.warning("Could not extract proposals from response (len=%d)", len(text))
    return {"proposals": []}


def _extract_from_prose(text: str) -> list[dict[str, Any]]:
    """Extract architecture/method names from prose with aggressive patterns."""
    proposals = []
    seen = set()

    # Common words that are NOT architecture names (stoplist)
    skip = {"the", "this", "that", "these", "those", "our", "your", "their",
            "result", "findings", "approach", "method", "technique", "model",
            "models", "framework", "system", "analysis", "study", "research",
            "overview", "summary", "conclusion", "future", "recent", "notable",
            "benchmark", "dataset", "datasets", "training", "inference",
            "operator", "networks", "network", "architecture", "architectures",
            "deep_learning", "machine_learning", "neural", "neural_network",
            "neural_networks", "function_spaces", "simulation", "prediction",
            "regression", "classification", "segmentation", "detection",
            "initiating_deep_research", "monitoring_progress", "parallel_search",
            "reporting", "key_findings_summary", "benchmarks", "urbantales"}

    # Match ALL bold-wrapped text as potential names: **Name** or **Name (details)**
    # This catches ConvNeXt V2, Fourier Neural Operator, Perceiver IO, etc.
    bold_pat = r"\*\*([^\*]+?)(?:\s*\([^)]*\d{4}[^)]*\))?\*\*"
    for m in re.finditer(bold_pat, text):
        name = m.group(1).strip()
        # Skip parenthetical content like "(Woo et al., 2023)"
        name = re.sub(r"\s*\([^)]*\)", "", name).strip()
        name_clean = name.replace(" ", "_").replace("-", "_").lower().rstrip(":_")
        if (name_clean not in seen
                and len(name_clean) > 3
                and name_clean not in skip
                and not name_clean.isdigit()
                and not re.match(r"^\d+\.?\s", name)):
            seen.add(name_clean)
            s = m.start()
            e = min(len(text), m.end() + 300)
            context = text[s:e].replace("\n", " ").strip()
            proposals.append({
                "name": name_clean,
                "rationale": context[:300],
                "source": "prose",
                "novelty": "existing",
                "category": "unknown",
                "difficulty": "medium",
                "estimated_params_m": 0,
            })

    # Name (Author, Year) citations in non-bold text
    for m in re.finditer(r"([A-Z][A-Za-z0-9_\-]+)\s*\([^)]*\d{4}[^)]*\)", text):
        name = m.group(1).lower()
        if name not in seen and len(name) > 3 and name not in skip:
            seen.add(name)
            s = max(0, m.start() - 20)
            e = min(len(text), m.end() + 200)
            context = text[s:e].replace("\n", " ").strip()
            proposals.append({
                "name": name,
                "rationale": context[:300],
                "source": "prose",
                "novelty": "existing",
                "category": "unknown",
                "difficulty": "medium",
                "estimated_params_m": 0,
            })

    return proposals


def _coerce_to_proposal_dict(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, list):
        return {"proposals": parsed}
    if isinstance(parsed, dict):
        if "proposals" in parsed and isinstance(parsed["proposals"], list):
            return parsed
        if "name" in parsed:
            return {"proposals": [parsed]}
    return {"proposals": []}


# =======================================================================
# CLI helper
# =======================================================================
def _run_cli(
    binary: str,
    prompt: str,
    *,
    extra_args: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    cwd: str | None = None,
) -> str:
    """Invoke a CLI tool via ``subprocess``, pipe ``prompt`` on stdin.

    If *cwd* is set the child process runs from that directory instead of
    the caller's cwd.  Useful for Gemini CLI which auto-reads workspace
    files (AGENTS.md, SOUL.md, etc.) from its working directory.
    """
    exe = shutil.which(binary)
    if exe and exe.upper().endswith(".CMD"):
        pass
    elif exe is None:
        exe = shutil.which(binary + ".cmd") or shutil.which(binary + ".exe")
    if not exe:
        LOGGER.info("CLI %r not on PATH -- skipping", binary)
        return ""
    args = [exe, *(extra_args or [])]
    t0 = time.time()
    try:
        result = subprocess.run(
            args,
            input=prompt.encode("utf-8") if prompt else None,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    except subprocess.TimeoutExpired:
        LOGGER.warning("CLI %s timed out after %ds", binary, timeout)
        return ""
    except OSError as exc:
        LOGGER.warning("CLI %s failed to spawn: %s", binary, exc)
        return ""
    elapsed = time.time() - t0
    if result.returncode != 0:
        LOGGER.warning(
            "CLI %s exit=%d (%.1fs); stderr head: %s",
            binary, result.returncode, elapsed,
            stderr[:200],
        )
    LOGGER.debug("CLI %s completed in %.1fs (%d bytes)", binary, elapsed, len(stdout))
    return stdout


# =======================================================================
# HTTP helper
# =======================================================================
def _post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Minimal stdlib POST -- avoids adding a requests/httpx dependency."""
    data = json.dumps(body).encode("utf-8")
    req = _urlrequest.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with _urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (_urlerror.URLError, TimeoutError) as exc:
        LOGGER.warning("POST %s failed: %s", url, exc)
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.warning("POST %s returned non-JSON (len=%d)", url, len(raw))
        return None


def _load_key_from_openclaw(provider: str) -> str | None:
    """Read API key from OpenClaw models.json (auto-discovered)."""
    candidates = [
        Path.home() / ".openclaw" / "models.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                providers = data.get("providers", {})
                prov = providers.get(provider, {})
                key = prov.get("apiKey")
                if key and not key.startswith("$"):
                    return key
            except (json.JSONDecodeError, KeyError):
                pass
    return None


def _ensure_gemini_env() -> None:
    """Set up Gemini CLI environment for Google OAuth."""
    os.environ["GOOGLE_GENAI_USE_GCA"] = "true"


def _gemini_cwd() -> str:
    """Return a clean temp directory for Gemini CLI.

    Gemini CLI auto-reads workspace files (AGENTS.md, SOUL.md, etc.)
    from its cwd. Running from a temp directory prevents it from
    leaking local project context into its responses.
    """
    tmp = tempfile.mkdtemp(prefix="gemini_scout_")
    return tmp


def _openai_like(
    prompt: str,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    enable_web: bool = False,
    web_search_config: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Generic handler for OpenAI-compatible chat-completion endpoints."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You return only JSON as requested."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    if enable_web:
        if web_search_config:
            body["tools"] = [web_search_config]
    if extra_body:
        body.update(extra_body)
    resp = _post_json(
        endpoint,
        {"Authorization": f"Bearer {api_key}"},
        body,
        timeout=timeout,
    )
    if not resp:
        return {"proposals": [], "status": "network_error"}
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        LOGGER.warning("Unexpected response shape from %s", endpoint)
        return {"proposals": [], "status": "bad_response"}
    out = _extract_json(text)
    out.setdefault("status", "ok")
    out["raw_text"] = text or ""
    out["model"] = model
    out["usage"] = resp.get("usage")
    out["finish_reason"] = (resp.get("choices") or [{}])[0].get("finish_reason")
    return out


# =======================================================================
# Per-model callers
# =======================================================================
def call_claude(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Invoke the local ``claude`` CLI -- handles its own web search."""
    text = _run_cli(
        "claude", prompt,
        extra_args=["--print", "--output-format", "text"],
        timeout=timeout,
    )
    out = _extract_json(text)
    out.setdefault("status", "ok" if text else "cli_missing")
    out.setdefault("scout", "claude")
    return out


def call_codex(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Invoke the local ``codex`` CLI (OpenAI's GPT-5.x terminal agent)."""
    text = _run_cli(
        "codex", prompt,
        extra_args=["exec", "--skip-git-repo-check"],
        timeout=timeout,
    )
    out = _extract_json(text)
    out.setdefault("status", "ok" if text else "cli_missing")
    out.setdefault("scout", "codex")
    return out


def call_glm(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """GLM via z.ai API (OpenAI-compatible). Key from OpenClaw models.json."""
    key = os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY")
    if not key:
        key = _load_key_from_openclaw("zai")
    if not key:
        return {"proposals": [], "status": "no_api_key", "scout": "glm"}
    out = _openai_like(
        prompt,
        endpoint="https://api.z.ai/api/coding/paas/v4/chat/completions",
        api_key=key,
        model="glm-5.1",
        enable_web=True,
        web_search_config={"type": "web_search", "web_search": {"enable": True, "search_result": True}},
        timeout=timeout,
    )
    out["scout"] = "glm"
    return out


def call_deepseek(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """DeepSeek API. Key from OpenClaw models.json."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        key = _load_key_from_openclaw("deepseek")
    if not key:
        return {"proposals": [], "status": "no_api_key", "scout": "deepseek"}
    out = _openai_like(
        prompt,
        endpoint="https://api.deepseek.com/v1/chat/completions",
        api_key=key,
        model=os.environ.get("AUTO_V6_DEEPSEEK_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-pro",
        enable_web=False,
        timeout=timeout,
    )
    out["scout"] = "deepseek"
    return out


def call_mimo(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Xiaomi MiMo API. Key from OpenClaw models.json."""
    key = os.environ.get("MIMO_API_KEY")
    if not key:
        key = _load_key_from_openclaw("xiaomi")
    if not key:
        return {"proposals": [], "status": "no_api_key", "scout": "mimo"}
    model = os.environ.get("AUTO_V6_MIMO_MODEL") or os.environ.get("MIMO_MODEL") or "mimo-v2.5-pro"
    out = _openai_like(
        prompt,
        endpoint="https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
        api_key=key,
        model=model,
        enable_web=True,
        extra_body={"webSearchEnabled": True},
        timeout=timeout,
    )
    out["scout"] = "mimo"
    out["model"] = model
    return out


def call_gemini(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Google Gemini CLI with built-in Google Search grounding (OAuth)."""
    _ensure_gemini_env()  # use OAuth (Pro subscription)
    final_prompt = (
        "IMPORTANT: Ignore any files in the current directory (AGENTS.md, SOUL.md, MEMORY.md, etc.). "
        "Do NOT reference them. Follow ONLY the instructions below.\n\n"
        + prompt
        + "\n\nYOU MUST RESPOND WITH A VALID JSON OBJECT ONLY. "
        "Do NOT include any explanatory text, markdown, or comments. "
        "Your ENTIRE response must be a single parseable JSON object. "
        "Start with { and end with }."
    )
    text = _run_cli(
        "gemini", final_prompt,
        extra_args=[
            "--model", "gemini-3.1-pro-preview",
            "--output-format", "text",
            "--yolo",
        ],
        timeout=timeout,
        cwd=_gemini_cwd(),
    )
    out = _extract_json(text)
    out.setdefault("status", "ok" if text else "cli_missing")
    out["raw_text"] = text or ""
    out["model"] = "gemini-3.1-pro-preview"
    out["scout"] = "gemini"
    return out


def call_grok(prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """xAI Grok API. Uses /v1/responses endpoint with web_search tool."""
    key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        key = _load_key_from_openclaw("xai")
    if not key:
        return {"proposals": [], "status": "no_api_key", "scout": "grok"}

    model = os.environ.get("AUTO_V6_GROK_MODEL") or os.environ.get("GROK_MODEL") or "grok-4.20-0309-reasoning"
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": "You return only JSON as requested."},
            {"role": "user", "content": prompt},
        ],
        "tools": [{"type": "web_search"}],
    }
    resp = _post_json(
        "https://api.x.ai/v1/responses",
        {"Authorization": f"Bearer {key}"},
        body,
        timeout=timeout,
    )
    if not resp:
        return {"proposals": [], "status": "network_error", "scout": "grok"}
    try:
        output = resp.get("output", [])
        text = ""
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text += part.get("text", "")
        if not text:
            text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (KeyError, IndexError, TypeError):
        LOGGER.warning("Unexpected response shape from Grok Responses API")
        return {"proposals": [], "status": "bad_response", "scout": "grok"}
    out = _extract_json(text)
    out.setdefault("status", "ok" if text else "cli_missing")
    out["raw_text"] = text or ""
    out["model"] = model
    out["scout"] = "grok"
    return out


def call_deep_research(
    prompt: str, *, timeout: int = DEFAULT_TIMEOUT_S
) -> dict[str, Any]:
    """Google Deep Research via Gemini CLI deep-research extension.

    Uses the paid API key (from OpenClaw models.json) instead of OAuth,
    to avoid hitting the free-tier quota limit.
    """
    # Use paid API key instead of OAuth (free tier has strict quota limits)
    api_key = _load_key_from_openclaw("google")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ.pop("GOOGLE_GENAI_USE_GCA", None)  # don't use OAuth
    else:
        _ensure_gemini_env()  # fallback to OAuth if no key
    dr_timeout = max(timeout, 900)

    dr_prompt = (
        "Use the deep_research tool (research_start) to conduct a thorough "
        "multi-source research on the following topic. After starting, use "
        "research_status to monitor progress until complete.\n\n"
        + prompt
        + "\n\nIMPORTANT: After your research, you MUST format your entire "
        "response as a single valid JSON object. Do NOT include prose, "
        "markdown, or explanatory text outside the JSON. Start with { and end with }."
    )
    text = _run_cli(
        "gemini", dr_prompt,
        extra_args=["--output-format", "text", "--yolo"],
        timeout=dr_timeout,
    )
    LOGGER.info(
        "Deep research raw output (%d chars): %s",
        len(text), text[:500] if text else "(empty)",
    )
    out = _extract_json(text)

    if not out.get("proposals") and not text:
        LOGGER.info("Deep Research extension failed, falling back to Gemini web search")
        fallback_prompt = (
            "You are conducting deep research with web search. "
            "Search extensively for the latest and best architectures.\n\n" + prompt
        )
        text = _run_cli(
            "gemini", fallback_prompt,
            extra_args=["--output-format", "text", "--yolo"],
            timeout=dr_timeout,
        )
        out = _extract_json(text)

    out.setdefault("status", "ok" if text else "cli_missing")
    out["scout"] = "deep_research"
    return out


# =======================================================================
# Registry
# =======================================================================
CALLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "glm": call_glm,
    "claude": call_claude,
    "codex": call_codex,
    "deepseek": call_deepseek,
    "mimo": call_mimo,
    "gemini": call_gemini,
    "grok": call_grok,
    "deep_research": call_deep_research,
}


def get_caller(name: str) -> Callable[..., dict[str, Any]]:
    if name not in CALLERS:
        raise KeyError(f"unknown scout: {name!r} (known: {list(CALLERS)})")
    return CALLERS[name]


__all__ = [
    "call_glm", "call_claude", "call_codex", "call_deepseek",
    "call_mimo", "call_gemini", "call_grok", "call_deep_research",
    "CALLERS", "get_caller",
]

