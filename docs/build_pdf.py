"""Render docs/proposal.md to a PDF for the application form (single PDF, A4).

Markdown -> HTML with a small converter (headings, paragraphs, lists, tables, images, bold,
italic, inline code, links; enough for the generated proposal), then Microsoft Word saves the
HTML as PDF through its COM interface (Windows with Word installed). Writes docs/proposal.pdf
and a copy in submission/. Run after docs/build_docs.py:

    python docs/build_pdf.py
"""
from __future__ import annotations

import html
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "proposal.md"
OUT_HTML = ROOT / "submission" / "proposal.html"
OUT_PDF = ROOT / "docs" / "proposal.pdf"

CSS = """
@page { size: 21cm 29.7cm; margin: 2cm 1.8cm 2cm 1.8cm; }
body { font-family: Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.3; color: #1f2328; }
h1 { font-size: 17pt; margin: 0 0 8pt 0; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt 0; border-bottom: 1px solid #c8ccd1; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
ul, ol { margin: 0 0 6pt 18pt; padding: 0; }
li { margin: 0 0 3pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font-size: 8pt; }
th, td { border: 0.5pt solid #9aa0a6; padding: 1pt 3pt; vertical-align: top; margin: 0; line-height: 1.1; }
th { background: #eef0f2; }
code { font-family: Consolas, monospace; font-size: 9pt; }
img { margin: 4pt 0 8pt 0; }
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def png_size(p: Path) -> tuple[int, int]:
    head = p.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{p} is not a PNG")
    return struct.unpack(">II", head[16:24])


def image(alt: str, src: str) -> str:
    """Give Word both dimensions: with only a width it keeps the native pixel height and squashes the plot."""
    p = (SRC.parent / src).resolve()
    w, h = png_size(p)
    width = 640 if w / h > 1.8 else 600          # px at 96 dpi; the A4 text block is about 655 px wide
    height = round(width * h / w)
    return (f'<p style="text-align:center"><img src="{p.as_uri()}" alt="{html.escape(alt)}" '
            f'width="{width}" height="{height}" style="width:{width * 0.75:.0f}pt;height:{height * 0.75:.0f}pt"></p>')


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, sep, body = cells[0], cells[1], cells[2:]
    align = ["right" if s.endswith(":") else "left" for s in sep]
    out = ["<table><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr>"]
    for r in body:
        out.append("<tr>" + "".join(f'<td style="text-align:{a}">{inline(c)}</td>' for c, a in zip(r, align)) + "</tr>")
    return "\n".join(out) + "</table>" + '<p style="margin:0;font-size:4pt">&nbsp;</p>'


def convert(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", ln)
        if m:
            out.append(image(m.group(1), m.group(2)))
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table(rows))
            continue
        m = re.match(r"^(\*|\d+\.)\s+", ln)
        if m:
            tag = "ul" if m.group(1) == "*" else "ol"
            items = []
            while i < len(lines) and lines[i].strip():
                mm = re.match(r"^(\*|\d+\.)\s+(.*)$", lines[i])
                if mm:
                    items.append(mm.group(2))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|\||!\[|\*\s|\d+\.\s)", lines[i]):
            para.append(lines[i].strip()); i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return out


def main():
    body = "\n".join(convert(SRC.read_text(encoding="utf-8")))
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Technical proposal</title>"
                        f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    ps = f"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {{
  $d = $w.Documents.Open('{OUT_HTML}', $false, $true)
  $d.ActiveWindow.View.Type = 3
  $d.PageSetup.PaperSize = 7
  $d.PageSetup.TopMargin = 56; $d.PageSetup.BottomMargin = 56
  $d.PageSetup.LeftMargin = 51; $d.PageSetup.RightMargin = 51
  foreach ($t in $d.Tables) {{
    $t.Range.ParagraphFormat.SpaceBefore = 0
    $t.Range.ParagraphFormat.SpaceAfter = 0
    $t.Range.ParagraphFormat.LineSpacingRule = 0
    $t.AutoFitBehavior(2)
  }}
  $d.SaveAs2('{OUT_PDF}', 17)
  $d.Close($false)
}} finally {{ $w.Quit() }}
"""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        sys.exit(f"Word conversion failed: {r.stderr.strip() or r.stdout.strip()}")
    shutil.copy(OUT_PDF, ROOT / "submission" / "technical_proposal.pdf")
    print(f"wrote {OUT_PDF.relative_to(ROOT)} and submission/technical_proposal.pdf")


if __name__ == "__main__":
    main()
