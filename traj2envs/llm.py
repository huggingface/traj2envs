"""The one LLM call the qa and build families make: a structured-output request to a model.

TRAJ2ENVS_MODEL picks the backend. A Claude id (`claude-opus-5-5`) goes to the Anthropic API
(ANTHROPIC_API_KEY); a Hub id (`Qwen/Qwen3-Coder-480B-A35B-Instruct`, optionally suffixed with
`:<provider>`, `:fastest` or `:cheapest`) goes to Hugging Face Inference Providers (HF_TOKEN).

The model only proposes (questions, contracts, tests); nothing it writes ships without passing
the checks in Docker. Responses are cached on disk, so re-running the pipeline is free and stable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pydantic

MODEL = os.environ.get("TRAJ2ENVS_MODEL") or "claude-opus-5-5"  # an empty value (unset shell variable) too
CACHE_DIR = Path(os.environ.get("TRAJ2ENVS_CACHE", Path.home() / ".cache" / "traj2envs" / "llm"))
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MAX_ATTEMPTS, RETRY_STATUSES = 6, {429, 500, 502, 503, 504}
# Reasoning models count their thinking against max_tokens: too small a budget ends in an empty answer.
MAX_OUTPUT_TOKENS = int(os.environ.get("TRAJ2ENVS_MAX_TOKENS", 32000))
REASONING_EFFORT = os.environ.get("TRAJ2ENVS_REASONING_EFFORT")  # e.g. low; sent as reasoning_effort when set
# Spending cap in USD over everything recorded in CACHE_DIR/usage, so it holds across reruns and across jobs
# sharing one cache (re-read every minute: parallel jobs can overshoot by about a minute of spending).
BUDGET_USD = float(os.environ["TRAJ2ENVS_BUDGET_USD"]) if os.environ.get("TRAJ2ENVS_BUDGET_USD") else None
# "input,output" USD per million tokens, for providers whose response doesn't report its cost.
PRICES = tuple(map(float, os.environ["TRAJ2ENVS_PRICES"].split(","))) if os.environ.get("TRAJ2ENVS_PRICES") else None
USAGE_RESCAN_SECONDS = 60


class BudgetExceeded(Exception):
    pass


class Spending:
    """Running total of what the calls recorded in CACHE_DIR/usage cost."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recorded, self.own_since_scan, self.scanned_at = 0.0, 0.0, None

    def total(self) -> float:
        with self.lock:
            if self.scanned_at is None or time.monotonic() - self.scanned_at > USAGE_RESCAN_SECONDS:
                self.recorded = sum(json.loads(path.read_text()).get("cost_usd") or 0.0
                                    for path in (CACHE_DIR / "usage").glob("*.json"))
                self.own_since_scan, self.scanned_at = 0.0, time.monotonic()
            return self.recorded + self.own_since_scan

    def record(self, usage: dict) -> None:
        cost = usage.get("cost")
        if cost is None and PRICES:
            cost = (usage.get("prompt_tokens", 0) * PRICES[0] + usage.get("completion_tokens", 0) * PRICES[1]) / 1e6
        if cost is None and BUDGET_USD is not None:
            raise BudgetExceeded("the provider doesn't report costs: set TRAJ2ENVS_PRICES to enforce the budget")
        (CACHE_DIR / "usage").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "usage" / f"{uuid.uuid4().hex}.json").write_text(json.dumps(
            {"model": MODEL, "cost_usd": cost, "time": time.time(), **usage}))
        with self.lock:
            self.own_since_scan += cost or 0.0


SPENDING = Spending()


def ask(system_prompt: str, user_prompt: str, response_schema):
    """Return a validated `response_schema` instance, or None if the model declined."""
    key = hashlib.sha256(f"{MODEL}\n{response_schema.__name__}\n{system_prompt}\n{user_prompt}".encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists():
        return response_schema.model_validate_json(cached.read_text())

    ask_model = ask_hf if "/" in MODEL else ask_anthropic
    parsed = ask_model(system_prompt, user_prompt, response_schema)
    if parsed is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(parsed.model_dump_json())
    return parsed


def ask_anthropic(system_prompt: str, user_prompt: str, response_schema):
    import anthropic  # optional dependency: pip install "traj2envs[llm]"

    response = anthropic.Anthropic().messages.parse(
        model=MODEL, max_tokens=16000, output_config={"effort": "high"},
        system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
        output_format=response_schema,
    )
    return None if response.stop_reason == "refusal" else response.parsed_output


def ask_hf(system_prompt: str, user_prompt: str, response_schema):
    """Inference Providers through the OpenAI-compatible router; HF_TOKEN (or `hf auth login`) pays."""
    from huggingface_hub import get_session, get_token

    headers = {"Authorization": f"Bearer {get_token()}"}
    if bill_to := os.environ.get("TRAJ2ENVS_BILL_TO"):  # an org to bill instead of the token's user
        headers["X-HF-Bill-To"] = bill_to
    payload = {
        "model": MODEL, "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": response_schema.__name__, "schema": response_schema.model_json_schema()}},
    }
    if REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT
    if BUDGET_USD is not None and (spent := SPENDING.total()) >= BUDGET_USD:
        raise BudgetExceeded(f"${spent:.2f} spent, budget ${BUDGET_USD:.2f}")
    for attempt in range(MAX_ATTEMPTS):  # rate limits and provider hiccups are routine in large batches
        try:
            response = get_session().post(HF_ROUTER_URL, headers=headers, timeout=900, json=payload)
            if response.status_code not in RETRY_STATUSES:
                break
        except OSError:  # connection errors and timeouts
            if attempt == MAX_ATTEMPTS - 1:
                raise
        time.sleep(min(2 ** attempt * 5, 120))
    response.raise_for_status()
    SPENDING.record(response.json().get("usage") or {})
    choice = response.json()["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return response_schema.model_validate_json(content)
    except pydantic.ValidationError as error:  # truncated, or a provider that ignores response_format: a decline
        failure = {"finish_reason": choice.get("finish_reason"), "usage": response.json().get("usage"),
                   "error": str(error)[:2000], "content": content}
        (CACHE_DIR / "declined").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "declined" / f"{hashlib.sha256(user_prompt.encode()).hexdigest()}.json").write_text(json.dumps(failure))
        return None
