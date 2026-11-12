from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


def test_no_misspelled_optimization_in_ui():
    assert "Optimalization" not in (APP_DIR / "app.py").read_text(encoding="utf-8")


def test_no_leftover_placeholder_comments():
    for path in APP_DIR.glob("*.py"):
        assert "cos tu bedzie" not in path.read_text(encoding="utf-8"), path.name
