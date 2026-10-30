"""Dump every notebook's code and stored outputs to plain text for review.

Usage (from the repository root):  python audit/00_extract_notebooks.py
Writes audit/notebooks/<name>.txt . Read-only: notebooks are never modified.
"""

import glob
import json
import os

os.makedirs("audit/notebooks", exist_ok=True)

for nb in sorted(glob.glob("*.ipynb")):
    with open(nb, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    out.append(f"# NOTEBOOK: {nb}")
    out.append(f"# nbformat={d.get('nbformat')}.{d.get('nbformat_minor')} cells={len(d['cells'])}")
    for i, c in enumerate(d["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            out.append(f"\n--- [{i}] MARKDOWN ---\n{src}")
        elif c["cell_type"] == "code":
            ec = c.get("execution_count")
            out.append(f"\n--- [{i}] CODE (exec {ec}) ---\n{src}")
            for o in c.get("outputs", []):
                t = o.get("output_type")
                if t == "stream":
                    txt = "".join(o.get("text", []))
                    out.append(f"    [stream] {txt[:1500]}")
                elif t in ("execute_result", "display_data"):
                    txt = "".join(o.get("data", {}).get("text/plain", []))
                    out.append(f"    [{t}] {txt[:1500]}")
                elif t == "error":
                    out.append(f"    [error] {o.get('ename')}: {o.get('evalue')}")
    name = os.path.join("audit", "notebooks", os.path.splitext(nb)[0] + ".txt")
    with open(name, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(name, len("\n".join(out)))
