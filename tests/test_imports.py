from pathlib import Path


def test_expected_top_level_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "trading_system").exists()
    assert (root / "scripts").exists()
    assert (root / "data").exists()
    assert (root / "artifacts").exists()
