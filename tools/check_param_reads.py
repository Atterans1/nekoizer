"""Proof that every tunable control is READ by the inference or output path (spec: 'search confirms this for every parameter name').

For each name in seedvc_params.ParamSpec, search the product code under audio/, engine/ and pipeline.py (not ui/, not tests, not
seedvc_params.py itself) for a read of that exact name: params.Get("name"), params["name"], params.get("name") or Snapshot()["name"].
A name that is only written (Set) or only appears in a comment does not count. Prints a table and exits 1 if any name has no read.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import seedvc_params as P  # noqa: E402

SEARCH = [ROOT / "pipeline.py"] + sorted((ROOT / "audio").glob("*.py")) + sorted((ROOT / "engine").glob("*.py"))


def Reads(name: str):
    pats = [
        re.compile(r"""\.Get\(\s*["']%s["']\s*\)""" % re.escape(name)),
        re.compile(r"""\[\s*["']%s["']\s*\]""" % re.escape(name)),
        re.compile(r"""\.get\(\s*["']%s["']""" % re.escape(name)),
    ]
    hits = []
    for f in SEARCH:
        if not f.exists():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if any(p.search(code) for p in pats):
                hits.append(f"{f.relative_to(ROOT).as_posix()}:{n}")
    return hits


def Main() -> int:
    missing = []
    print(f"{'parameter':18} reads  first locations")
    for name in P.ParamSpec:
        hits = Reads(name)
        print(f"{name:18} {len(hits):5}  {', '.join(hits[:3]) or 'NONE'}")
        if not hits:
            missing.append(name)
    print("files searched:", ", ".join(f.relative_to(ROOT).as_posix() for f in SEARCH if f.exists()))
    print("MISSING:", missing or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(Main())
