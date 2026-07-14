"""Tests for the analyzer's transient-error retry loop (a one-shot catalyst must survive a
429/timeout/5xx blip), without any network/client construction."""
import asyncio
import types

import pytest

from hlbot.analysis import analyzer as analyzer_mod
from hlbot.analysis.analyzer import Analyzer


class Boom(Exception):
    pass


def _analyzer(max_retries=4):
    a = object.__new__(Analyzer)  # bypass __init__ (no real client / api key)
    a.cfg = types.SimpleNamespace(max_retries=max_retries, retry_base_delay=0.0)
    return a


def _client(create):
    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "_is_retryable", lambda e: isinstance(e, Boom))
    calls = {"n": 0}

    async def create(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom()
        return "ok"

    a = _analyzer(max_retries=4)
    a.client = _client(create)
    assert asyncio.run(a._create(model="x")) == "ok"
    assert calls["n"] == 3


def test_non_retryable_raises_immediately(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "_is_retryable", lambda e: False)
    calls = {"n": 0}

    async def create(**_kw):
        calls["n"] += 1
        raise Boom()

    a = _analyzer(max_retries=4)
    a.client = _client(create)
    with pytest.raises(Boom):
        asyncio.run(a._create())
    assert calls["n"] == 1  # no retry on a non-retryable error


def test_exhausts_retries_then_raises(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "_is_retryable", lambda e: True)
    calls = {"n": 0}

    async def create(**_kw):
        calls["n"] += 1
        raise Boom()

    a = _analyzer(max_retries=3)
    a.client = _client(create)
    with pytest.raises(Boom):
        asyncio.run(a._create())
    assert calls["n"] == 3  # exactly max_retries attempts


def test_is_retryable_status_codes():
    # Real APIStatusError instances: only 408/409/429/5xx are retryable; 4xx (e.g. 400) is not.
    import httpx
    from anthropic import APIConnectionError, APIStatusError

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for code, expect in [(429, True), (500, True), (503, True), (408, True),
                         (400, False), (404, False)]:
        exc = APIStatusError("e", response=httpx.Response(code, request=req), body=None)
        assert analyzer_mod._is_retryable(exc) is expect

    assert analyzer_mod._is_retryable(APIConnectionError(message="x", request=req)) is True
    assert analyzer_mod._is_retryable(ValueError("nope")) is False
