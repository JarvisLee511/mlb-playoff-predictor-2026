"""Print done=true once the configured season is complete; the workflows read it.

Stdlib only, because it runs before any dependencies are installed: a finished season then
costs a few seconds of runner time instead of an install, a retrain and a commit. Bumping
CURRENT_SEASON in src/config.py re-opens the gate for the next year.
"""
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
config = (root / "src" / "config.py").read_text(encoding="utf-8")
current = int(re.search(r"CURRENT_SEASON\s*=\s*(\d+)", config).group(1))
path = root / "docs" / "data" / "season.json"
state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
done = state.get("season") == current and state.get("status") == "complete"
print(f"done={'true' if done else 'false'}")
