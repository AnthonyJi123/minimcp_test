"""Compile paper/main.tex on Modal (no local TeX toolchain).

Writes the built PDF + log back to the gate-data volume under
build/, so it can be fetched with `modal volume get`.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _build_pdf.py::build
"""
import modal

app = modal.App("paper-build")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.from_registry("texlive/texlive:latest",
                                 add_python="3.11")
       .add_local_dir("paper", "/paper", copy=True))


@app.function(image=img, volumes={"/data": vol}, timeout=60 * 20)
def build():
    import os
    import shutil
    import subprocess

    os.chdir("/paper")

    def run(cmd):
        p = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True)
        return p.stdout + p.stderr

    logs = []
    for cmd in ("pdflatex -interaction=nonstopmode main.tex",
                "bibtex main",
                "pdflatex -interaction=nonstopmode main.tex",
                "pdflatex -interaction=nonstopmode main.tex"):
        out = run(cmd)
        logs.append(f"$ {cmd}\n{out[-3000:]}")

    ok = os.path.exists("/paper/main.pdf")
    os.makedirs("/data/build", exist_ok=True)
    if ok:
        shutil.copy("/paper/main.pdf", "/data/build/main.pdf")
    with open("/data/build/build.log", "w") as fh:
        fh.write("\n\n".join(logs))
    vol.commit()

    # surface the things that matter: errors, warnings, page count
    tail = run("grep -nE '^!|Undefined control|Overfull|LaTeX Warning: "
               "(Citation|Reference)|Output written' /paper/main.log "
               "| tail -40")
    print(tail)
    print(f"PDF built: {ok}, "
          f"size={os.path.getsize('/paper/main.pdf') if ok else 0}")


@app.function(image=img, volumes={"/data": vol}, timeout=300)
def where():
    """Page numbers of the labels we care about (needs a prior build)."""
    import subprocess
    for cmd in ("pdflatex -interaction=nonstopmode main.tex",):
        subprocess.run(cmd, shell=True, cwd="/paper",
                       capture_output=True, text=True)
    aux = open("/paper/main.aux").read()
    import re
    for lab in ("tab:floor", "app:duplexval"):
        m = re.search(r"\\newlabel\{" + lab + r"\}\{\{([^}]*)\}\{(\d+)\}",
                      aux)
        print(f"{lab}: {m.group(1) if m else '?'} on page "
              f"{m.group(2) if m else '?'}")
