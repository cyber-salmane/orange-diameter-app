import importlib
import os


def test_app_base_url_default(monkeypatch):
    monkeypatch.delenv("APP_BASE_URL", raising=False)
    config = importlib.reload(importlib.import_module("config"))
    assert config.APP_BASE_URL == "http://localhost:7860"
