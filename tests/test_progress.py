from io import StringIO
import sys

import pytest

from bench.progress import progress_bar


@pytest.mark.parametrize("enabled, terminal, visible", [
    (None, False, False), (None, True, True),
    (False, False, False), (False, True, False),
    (True, False, True), (True, True, True),
])
def test_progress_respects_terminal_and_explicit_override(monkeypatch, enabled, terminal, visible):
    stream = StringIO()
    stream.isatty = lambda: terminal
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setenv("TERM", "xterm-256color")
    stdout = sys.stdout
    with progress_bar(total=3, desc="extract", unit="image", enabled=enabled) as bar:
        assert bar.disable is not visible
        bar.update(3)
    assert bool(stream.getvalue()) is visible
    assert bar.disable
    assert sys.stdout is stdout and sys.stderr is stream


def test_progress_closes_on_failure(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    with pytest.raises(RuntimeError, match="failed stage"):
        with progress_bar(total=3, desc="fit", unit="update", enabled=True) as bar:
            bar.update(1)
            raise RuntimeError("failed stage")
    assert bar.disable
    assert not bar._prog.live.is_started


def test_progress_skips_empty_cached_stage(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    with progress_bar(total=0, desc="extract", unit="image", enabled=True) as bar:
        assert bar.disable
    assert not stream.getvalue()
