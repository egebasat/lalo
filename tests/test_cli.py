import logging

from agent_bot.cli import setup_logging


def test_setup_logging_applies_the_level_to_agent_bot_only(monkeypatch):
    monkeypatch.setattr(logging.getLogger("agent_bot"), "level", logging.NOTSET)
    setup_logging("DEBUG")
    assert logging.getLogger("agent_bot.inference").isEnabledFor(logging.DEBUG)
    # Libraries such as httpx log every request at INFO; keep them out of the terminal.
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
