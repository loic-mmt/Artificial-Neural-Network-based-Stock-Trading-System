"""Launch the optional dashboard or inspect its catalogue without Streamlit."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from trading_system.paths import artifacts_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=artifacts_root())
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true", help="Do not open a browser automatically.")
    parser.add_argument("--list", action="store_true", help="Print the discovered runs without launching the UI.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum catalogue rows printed by --list.")
    args = parser.parse_args(argv)
    root = args.artifacts_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Artifact directory does not exist: {root}")
    if args.limit <= 0 or not 1 <= args.port <= 65535:
        parser.error("--limit must be positive and --port must be between 1 and 65535.")
    if args.list:
        from .catalog import catalog_frame, discover_runs

        catalog = discover_runs(root)
        frame = catalog_frame(catalog.records)
        columns = [name for name in ("run_id", "label", "family", "status") if name in frame]
        print(frame[columns].head(args.limit).to_string(index=False) if len(frame) else "No saved runs found.")
        print(f"\n{len(catalog.records)} runs; {len(catalog.issues)} discovery issues.")
        for issue in catalog.issues[:10]:
            print(issue, file=sys.stderr)
        return 0
    if importlib.util.find_spec("streamlit") is None:
        parser.error('Dashboard dependency missing. Install: python -m pip install -e ".[visualization]"')
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
        "--server.address", "127.0.0.1", "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false",
        "--theme.base", "light", "--theme.primaryColor", "#165dff",
        "--theme.backgroundColor", "#f7f8fa", "--theme.secondaryBackgroundColor", "#ffffff",
        "--theme.textColor", "#101318",
        "--", "--artifacts-root", str(root),
    ]
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
