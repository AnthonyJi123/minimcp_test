"""Sanity check: LaTeX tabular row cell counts vs declared column spec."""
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent.parent / "paper" / "sections"
PAT = re.compile(
    r"\\label\{(tab:transfer|tab:nvda-remix)\}(.*?)\\end\{tabular\}", re.S)

for f in ["transfer.tex", "appendix.tex"]:
    src = (BASE / f).read_text(encoding="utf-8")
    for m in PAT.finditer(src):
        lbl, chunk = m.group(1), m.group(2)
        t = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*)", chunk, re.S)
        spec, body = t.group(1), t.group(2)
        ncol = len(re.findall(r"[lcr]", re.sub(r"p\{[^}]*\}", "p", spec)))
        print(f"== {lbl} ({f}): declared {ncol} cols")
        bad = 0
        for line in body.split("\\\\"):
            line = line.strip()
            if "&" not in line:
                continue
            if "multicolumn" in line:
                span = int(re.search(r"\\multicolumn\{(\d+)\}", line).group(1))
                got = span + line.count("&")
            else:
                got = line.count("&") + 1
            if got != ncol:
                bad += 1
                print(f"   MISMATCH ({got}): {line[:72]}")
        print("   OK" if bad == 0 else f"   {bad} bad rows")
