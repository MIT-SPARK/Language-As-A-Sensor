"""OpenAI-compatible chat client with a content-addressed JSON response cache.

Every LLM row in the paper is reproducible from the cache shipped in the
artifacts (see docs/reproduce.md). Set ``LANGSENSOR_OFFLINE=1`` to turn a cache
miss into an error instead of an API call — that is how a reproduction run
proves it made no calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

OLLAMA_BASE_URL = "http://localhost:11434/v1"

class OfflineCacheMiss(RuntimeError):
    pass


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class JsonCache:
    """``<dir>/<key>.json``; a no-op when *directory* is None."""

    def __init__(self, directory: str | Path | None) -> None:
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        if self.dir is None or not (self.dir / f"{key}.json").exists():
            return None
        try:
            return json.loads((self.dir / f"{key}.json").read_text())
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict) -> None:
        if self.dir is not None:
            (self.dir / f"{key}.json").write_text(json.dumps(value, indent=2))


class ChatClient:
    """Thin wrapper over ``openai.OpenAI().chat.completions`` (OpenAI or a local Ollama server)."""

    def __init__(self, provider: str = "openai", model: str = "gpt-5.2", base_url: str | None = None,
                 api_key: str | None = None, temperature: float | None = None, seed: int | None = 42) -> None:
        self.provider = provider.strip().lower()
        self.model = model
        self.temperature = temperature
        self.seed = seed
        if self.provider == "ollama":
            self.base_url, self.api_key = base_url or OLLAMA_BASE_URL, api_key or "ollama"
        else:
            self.base_url, self.api_key = base_url, api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    def complete(self, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        """Return (response text, token usage)."""
        if os.environ.get("LANGSENSOR_OFFLINE") == "1":
            raise OfflineCacheMiss(f"cache miss for {self.model} with LANGSENSOR_OFFLINE=1")
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url.rstrip("/")
            self._client = OpenAI(**kwargs)

        kwargs: dict = {"model": self.model, "messages": messages}
        temperature = self.temperature if temperature is None else temperature
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.seed is not None and self.provider == "openai":
            kwargs["seed"] = self.seed
        resp = self._client.chat.completions.create(**kwargs)
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return (resp.choices[0].message.content or "").strip(), usage


# ── Parsing LLM output ────────────────────────────────────────────────────────

MIN_EIGENVAL = 0.01    # m², floor when projecting an LLM covariance onto the PSD cone
DEFAULT_SIGMA = 0.5    # m, isotropic fallback


def extract_json_object(text: str) -> dict | None:
    """The first balanced ``{...}`` in *text*, parsed, or None."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def psd_cholesky(sigma, min_eigval: float = MIN_EIGENVAL) -> np.ndarray:
    """Cholesky factor of the nearest PSD matrix (symmetrise, clip eigenvalues)."""
    S = np.asarray(sigma, dtype=np.float64).reshape(3, 3)
    S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    S = (eigvecs * np.clip(eigvals, min_eigval, None)) @ eigvecs.T
    return np.linalg.cholesky(0.5 * (S + S.T)).astype(np.float32)


def parse_mu_sigma(obj: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """``{"mu": [3], "sigma": [3x3] or [3]}`` -> (mu, L), or None if unusable."""
    mu_raw = obj.get("mu") or obj.get("mean") or obj.get("center")
    sig_raw = obj.get("sigma") or obj.get("cov") or obj.get("covariance")
    if mu_raw is None or sig_raw is None:
        return None
    try:
        mu = np.asarray(mu_raw, dtype=np.float32).reshape(3)
        sig = np.asarray(sig_raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if sig.shape == (3,):
        sig = np.diag(np.clip(sig, MIN_EIGENVAL, None))
    if sig.shape != (3, 3):
        return None
    try:
        return mu, psd_cholesky(sig)
    except np.linalg.LinAlgError:
        return None


def as_tensors(mu: np.ndarray, L: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(np.asarray(mu, dtype=np.float32)), torch.from_numpy(np.asarray(L, dtype=np.float32))
