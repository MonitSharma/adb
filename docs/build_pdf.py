"""Render docs/proposal.md to a PDF for the application form (single PDF, A4).

Markdown -> HTML with a small converter (headings, paragraphs, lists, tables, images, bold,
italic, inline code, links; enough for the generated proposal), then Microsoft Word saves the
HTML as PDF through its COM interface (Windows with Word installed). Writes docs/proposal.pdf
Run after docs/build_docs.py:

    python docs/build_pdf.py
"""
from __future__ import annotations

import html
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "proposal.md"
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
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font-size: 8pt; width: 100%; }
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


TEXT_WIDTH_PT = 493                               # A4 minus the 51 pt side margins set below
COLUMN_WIDTHS: list[list[float]] = []             # one list per table, in document order


def column_widths(cells: list[list[str]]) -> list[float]:
    """Share the text width between columns by the length of their longest cell (clamped), so short
    numeric columns stay narrow and prose columns get room. Word's own AutoFit is unreliable on HTML
    imports: it can lay every column out at a fixed 24 pt."""
    plain = [[re.sub(r"[*`]", "", c) for c in r] for r in cells]
    n = max(len(r) for r in plain)
    w = [min(max(max((len(r[j]) for r in plain if j < len(r)), default=4), 6), 45) for j in range(n)]
    return [round(TEXT_WIDTH_PT * x / sum(w), 1) for x in w]


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, sep, body = cells[0], cells[1], cells[2:]
    COLUMN_WIDTHS.append(column_widths([head] + body))
    align = ["right" if s.endswith(":") else "left" for s in sep]
    out = ['<table width="100%"><tr>' + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr>"]
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
    OUT_HTML = Path(tempfile.mkdtemp(prefix="proposal_")) / "proposal.html"
    OUT_HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Proposal</title>"
                        f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    widths = ", ".join("@(" + ", ".join(f"{x:g}" for x in w) + ")" for w in COLUMN_WIDTHS)
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
  $widths = @({widths})
  if ($d.Tables.Count -ne $widths.Count) {{ throw "expected $($widths.Count) tables, Word found $($d.Tables.Count)" }}
  $ti = 0
  foreach ($t in $d.Tables) {{
    $t.Range.ParagraphFormat.SpaceBefore = 0
    $t.Range.ParagraphFormat.SpaceAfter = 0
    $t.Range.ParagraphFormat.LineSpacingRule = 0
    $t.AllowAutoFit = $false
    $cw = $widths[$ti]; $ti++
    for ($j = 1; $j -le $t.Columns.Count; $j++) {{ $t.Columns.Item($j).Width = [single]$cw[$j - 1] }}
    # keep each table on one page: no row splits, every row tied to the next, and the
    # paragraph that introduces the table tied to it as well
    $t.Rows.AllowBreakAcrossPages = 0
    $t.Range.ParagraphFormat.KeepWithNext = -1
    $t.Rows.Last.Range.ParagraphFormat.KeepWithNext = 0
    $prev = $t.Range.Previous(4, 1)
    if ($prev -ne $null) {{ $prev.ParagraphFormat.KeepWithNext = -1 }}
  }}
  # never leave a heading alone at the foot of a page
  foreach ($p in $d.Paragraphs) {{ if ($p.OutlineLevel -lt 10) {{ $p.KeepWithNext = -1 }} }}
  $d.SaveAs2('{OUT_PDF}', 17)
  $d.Close($false)
}} finally {{ $w.Quit() }}
"""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True)
    if r.returncode != 0 or not OUT_PDF.exists():
        sys.exit(f"Word conversion failed: {r.stderr.strip() or r.stdout.strip()}")
    shutil.rmtree(OUT_HTML.parent, ignore_errors=True)
    print(f"wrote {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
