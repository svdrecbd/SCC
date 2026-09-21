#!/usr/bin/env python3
"""Render the editable whitepaper with a typeset body and vector cover.

Requires ReportLab, pypdf, Pandoc, and XeLaTeX with TeX Gyre fonts.
Intermediate files remain in the supplied work directory.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


DOCUMENT_TITLE = "Safety–Capability Coupling Whitepaper"
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "deliverables/scc-whitepaper/Safety_Capability_Coupling_Whitepaper.md"
OUTPUT = REPOSITORY / "output/pdf/Safety_Capability_Coupling_Whitepaper.pdf"
LATEX_HEADER = r"""
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{needspace}
\definecolor{DocumentNavy}{HTML}{18344A}
\definecolor{DocumentTeal}{HTML}{246B75}
\titleformat{\section}{\sffamily\Large\bfseries\color{DocumentNavy}}{}{0pt}{\needspace{6\baselineskip}}
\titleformat{\subsection}{\sffamily\normalsize\bfseries\color{DocumentNavy}}{}{0pt}{\needspace{4\baselineskip}}
\titlespacing*{\section}{0pt}{20pt}{9pt}
\titlespacing*{\subsection}{0pt}{13pt}{5pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\scriptsize\color{DocumentNavy}SAFETY–CAPABILITY COUPLING}
\fancyhead[R]{\sffamily\scriptsize WHITEPAPER · VERSION 1.0}
\fancyfoot[L]{\sffamily\scriptsize 20 September 2026}
\fancyfoot[R]{\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\headheight}{15pt}
\setlength{\emergencystretch}{3em}
\setcounter{page}{2}
\AtBeginDocument{\hypersetup{pdftitle={Safety–Capability Coupling Whitepaper}}}
\widowpenalty=10000
\clubpenalty=10000
"""


def write_cover(destination: Path) -> None:
    """Create the cover using embedded type and vector drawing operations."""
    font_directory = Path(__import__("reportlab").__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("DocumentSans", str(font_directory / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("DocumentSansBold", str(font_directory / "VeraBd.ttf")))
    document = canvas.Canvas(str(destination), pagesize=(612, 792), invariant=1)
    document.setTitle(DOCUMENT_TITLE)
    document.setFillColor(HexColor("#18344A"))
    document.rect(56, 683, 48, 5, fill=1, stroke=0)
    document.setFont("DocumentSans", 10)
    document.drawString(56, 654, "RESEARCH WHITEPAPER")
    title_style = ParagraphStyle("DocumentTitle", fontName="DocumentSansBold", fontSize=30,
                                 leading=38, textColor=HexColor("#18344A"))
    title = Paragraph("Safety–Capability<br/>Coupling Whitepaper", title_style)
    _, title_height = title.wrap(500, 160)
    title.drawOn(document, 56, 615 - title_height)
    body_style = ParagraphStyle("CoverDescription", fontName="DocumentSans", fontSize=12,
                                leading=19, textColor=HexColor("#3B4E5D"))
    description = Paragraph("A mathematical account of the intended mechanism,<br/>"
                            "established results, and remaining construction requirements.", body_style)
    _, description_height = description.wrap(495, 120)
    description.drawOn(document, 56, 478 - description_height)
    document.setStrokeColor(HexColor("#B8C9CD"))
    document.line(56, 355, 556, 355)
    document.setFillColor(HexColor("#18344A"))
    document.setFont("DocumentSansBold", 11)
    document.drawString(56, 324, "Version 1.0")
    document.setFont("DocumentSans", 11)
    document.drawString(56, 301, "20 September 2026")
    note_style = ParagraphStyle("CoverStatus", fontName="DocumentSans", fontSize=10,
                                leading=16, textColor=HexColor("#3B4E5D"))
    note = Paragraph("Evidence through LN-239. A synthesis of an active research program, "
                     "prepared as a precursor to a formal paper. The record establishes "
                     "neither a working intrinsic SCC mechanism nor a general impossibility theorem.", note_style)
    _, note_height = note.wrap(470, 120)
    note.drawOn(document, 56, 242 - note_height)
    document.setFont("DocumentSans", 9)
    document.drawString(56, 57, "Safety–Capability Coupling research program")
    document.drawRightString(556, 57, "1")
    document.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-directory", type=Path, required=True)
    arguments = parser.parse_args()
    work_directory = arguments.work_directory.resolve()
    work_directory.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE.read_text()
    body = source_text.split("<!-- DOCUMENT BODY -->", 1)[1].strip()
    body_path = work_directory / "document_body.md"
    body_path.write_text(body + "\n")
    header_path = work_directory / "document_header.tex"
    header_path.write_text(LATEX_HEADER)
    latex_path = work_directory / "document_body.tex"
    subprocess.run([
        "pandoc", str(body_path), "--standalone", "--from=markdown+raw_tex", "--to=latex",
        "--table-of-contents", "--toc-depth=1", "--include-in-header=" + str(header_path),
        "-V", "documentclass=article", "-V", "fontsize=11pt", "-V", "papersize=letter",
        "-V", "geometry:margin=0.8in", "-V", "mainfont=texgyrepagella-regular.otf",
        "-V", "mainfontoptions=BoldFont=texgyrepagella-bold.otf,ItalicFont=texgyrepagella-italic.otf,BoldItalicFont=texgyrepagella-bolditalic.otf",
        "-V", "sansfont=texgyreheros-regular.otf",
        "-V", "sansfontoptions=BoldFont=texgyreheros-bold.otf,ItalicFont=texgyreheros-italic.otf",
        "-V", "monofont=texgyrecursor-regular.otf",
        "-V", "mathfont=texgyrepagella-math.otf", "-V", "colorlinks=true",
        "-V", "linkcolor=DocumentTeal", "-V", "urlcolor=DocumentTeal",
        "-V", "toc-title=Contents", "-o", str(latex_path)
    ], check=True)
    latex_text = latex_path.read_text().replace("\\tableofcontents\n", "\\tableofcontents\n\\clearpage\n")
    def format_table(match: re.Match) -> str:
        table = match.group(0)
        if table.count("\\real{0.5000}") == 2:
            return table.replace("\\real{0.5000}", "\\real{0.3000}", 1).replace("\\real{0.5000}", "\\real{0.7000}", 1)
        if table.count("\\real{0.3333}") == 3:
            for width in ("0.2900", "0.4900", "0.2200"):
                table = table.replace("\\real{0.3333}", "\\real{" + width + "}", 1)
        return table
    latex_text = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", format_table, latex_text, flags=re.S)
    latex_text = latex_text.replace("\\section{References}", "\\clearpage\n\\section{References}")
    latex_path.write_text(latex_text)
    with (work_directory / "typesetting_output.txt").open("w") as output_stream:
        for _ in range(3):
            subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            str(latex_path)], cwd=work_directory, stdout=output_stream,
                           stderr=subprocess.STDOUT, check=True)
    cover_path = work_directory / "document_cover.pdf"
    write_cover(cover_path)
    writer = PdfWriter()
    writer.append(str(cover_path))
    writer.append(str(work_directory / "document_body.pdf"))
    writer.add_metadata({"/Title": DOCUMENT_TITLE,
                         "/Subject": "Mathematical results and construction requirements through LN-239",
                         "/Creator": "SCC whitepaper renderer"})
    with OUTPUT.open("wb") as output_stream:
        writer.write(output_stream)
    receipt = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "output": str(OUTPUT), "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
               "pages": len(PdfReader(OUTPUT).pages)}
    (work_directory / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
