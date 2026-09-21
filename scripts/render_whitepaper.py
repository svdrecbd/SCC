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
import xml.etree.ElementTree as ET
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


DOCUMENT_TITLE = "Safety–Capability Coupling Program"
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "deliverables/scc-whitepaper/Safety_Capability_Coupling_Whitepaper.md"
OUTPUT = REPOSITORY / "output/pdf/Safety_Capability_Coupling_Whitepaper.pdf"
ILLUSTRATION = REPOSITORY / "deliverables/geometric-studies/projected_sections.svg"
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
\AtBeginDocument{\hypersetup{pdftitle={Safety–Capability Coupling Program}}}
\widowpenalty=10000
\clubpenalty=10000
"""


def draw_cover_illustration(document) -> None:
    """Place the approved SVG polylines as native PDF vectors."""
    root = ET.parse(ILLUSTRATION).getroot()
    paths = []
    for element in root.findall("{http://www.w3.org/2000/svg}polyline"):
        coordinates = [tuple(map(float, point.split(","))) for point in element.attrib["points"].split()]
        paths.append((coordinates, float(element.attrib["stroke-width"]), element.attrib["stroke"]))
    minimum_x = min(x for coordinates, _, _ in paths for x, y in coordinates)
    maximum_x = max(x for coordinates, _, _ in paths for x, y in coordinates)
    minimum_y = min(y for coordinates, _, _ in paths for x, y in coordinates)
    maximum_y = max(y for coordinates, _, _ in paths for x, y in coordinates)
    allocation_scale = 480 / (maximum_x - minimum_x)
    allocation_height = (maximum_y - minimum_y) * allocation_scale
    scale = 0.8 * allocation_scale
    left = 66 + 0.1 * 480
    bottom = 235 + 0.1 * allocation_height
    document.saveState()
    document.setLineCap(1)
    document.setLineJoin(1)
    for coordinates, width, color in paths:
        path = document.beginPath()
        for index, (x, y) in enumerate(coordinates):
            point = (left + (x - minimum_x) * scale, bottom + (maximum_y - y) * scale)
            if index == 0:
                path.moveTo(*point)
            else:
                path.lineTo(*point)
        document.setStrokeColor(HexColor(color))
        document.setLineWidth(width * scale)
        document.drawPath(path, fill=0, stroke=1)
    document.restoreState()


def write_cover(destination: Path) -> None:
    """Create the cover using embedded type and vector drawing operations."""
    font_directory = Path(__import__("reportlab").__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("DocumentSans", str(font_directory / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("DocumentSansBold", str(font_directory / "VeraBd.ttf")))
    document = canvas.Canvas(str(destination), pagesize=(612, 792), invariant=1)
    document.setTitle(DOCUMENT_TITLE)
    document.setFillColor(HexColor("#18344A"))
    document.rect(56, 743, 48, 5, fill=1, stroke=0)
    document.setFont("DocumentSans", 10)
    document.drawString(56, 714, "RESEARCH WHITEPAPER")
    title_style = ParagraphStyle("DocumentTitle", fontName="DocumentSansBold", fontSize=30,
                                 leading=38, textColor=HexColor("#18344A"))
    title = Paragraph("Safety–Capability<br/>Coupling Program", title_style)
    _, title_height = title.wrap(500, 160)
    title.drawOn(document, 56, 675 - title_height)
    body_style = ParagraphStyle("CoverDescription", fontName="DocumentSans", fontSize=12,
                                leading=19, textColor=HexColor("#3B4E5D"))
    description = Paragraph("A mathematical account of the intended mechanism,<br/>"
                            "established results, and remaining construction requirements.", body_style)
    _, description_height = description.wrap(495, 120)
    description.drawOn(document, 56, 562 - description_height)
    draw_cover_illustration(document)
    document.setStrokeColor(HexColor("#B8C9CD"))
    document.line(56, 210, 556, 210)
    document.setFillColor(HexColor("#18344A"))
    document.setFont("DocumentSansBold", 11)
    document.drawString(56, 187, "Version 1.0")
    document.setFont("DocumentSans", 11)
    document.drawRightString(556, 187, "20 September 2026")
    note_style = ParagraphStyle("CoverStatus", fontName="DocumentSans", fontSize=10,
                                leading=16, textColor=HexColor("#3B4E5D"))
    note = Paragraph("Evidence through LN-239. A synthesis of an active research program, "
                     "prepared as a precursor to a formal paper. The record establishes "
                     "neither a working intrinsic SCC mechanism nor a general impossibility theorem.", note_style)
    _, note_height = note.wrap(470, 120)
    note.drawOn(document, 56, 152 - note_height)
    document.setFont("DocumentSans", 9)
    document.drawRightString(556, 57, "1")
    document.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--columns", type=int, choices=(1, 2), default=1)
    arguments = parser.parse_args()
    work_directory = arguments.work_directory.resolve()
    work_directory.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT if arguments.columns == 1 else OUTPUT.with_name("Safety_Capability_Coupling_Whitepaper_Two_Columns.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE.read_text()
    body = source_text.split("<!-- DOCUMENT BODY -->", 1)[1].strip()
    body_path = work_directory / "document_body.md"
    body_path.write_text(body + "\n")
    header_path = work_directory / "document_header.tex"
    header = LATEX_HEADER
    if arguments.columns == 2:
        header = header.replace(r"\sffamily\Large", r"\sffamily\large")
        header += r"""
\usepackage{multicol}
\usepackage{balance}
\setlength{\columnsep}{22pt}
\setlength{\parskip}{5pt plus 1pt minus 1pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\raggedbottom
"""
    header_path.write_text(header)
    latex_path = work_directory / "document_body.tex"
    subprocess.run([
        "pandoc", str(body_path), "--standalone", "--from=markdown+raw_tex", "--to=latex",
        "--table-of-contents", "--toc-depth=1", "--include-in-header=" + str(header_path),
        "-V", "documentclass=article", "-V", "fontsize=" + ("10pt" if arguments.columns == 2 else "11pt"), "-V", "papersize=letter",
        "-V", "geometry:margin=" + ("0.7in" if arguments.columns == 2 else "0.8in"), "-V", "mainfont=texgyrepagella-regular.otf",
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
    if arguments.columns == 2:
        latex_text = format_two_columns(latex_text)
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
    with output_path.open("wb") as output_stream:
        writer.write(output_stream)
    receipt = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "output": str(output_path), "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
               "pages": len(PdfReader(output_path).pages), "columns": arguments.columns,
               "illustration": str(ILLUSTRATION),
               "illustration_sha256": hashlib.sha256(ILLUSTRATION.read_bytes()).hexdigest()}
    (work_directory / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def format_two_columns(latex_text: str) -> str:
    """Reflow the same content, retaining full width for the evidence table."""
    latex_text = latex_text.replace("\\tableofcontents\n\\clearpage\n}", "\\tableofcontents\n}\n\\clearpage\n\\begin{multicols}{2}")
    latex_text = latex_text.replace("\\newpage", "\\end{multicols}\n\\twocolumn", 1)
    latex_text = latex_text.replace(r"\section{13.", r"\balance\section{13.")
    latex_text = re.sub(r"\\newpage\s+(?=\\section\{Appendix A\.)", lambda _: "\\clearpage\n\\nobalance\n", latex_text)
    appendix_position = latex_text.index(r"\section{Appendix B.")
    preceding = latex_text[:appendix_position]
    following = latex_text[appendix_position:]

    def column_table(match: re.Match) -> str:
        table = match.group(0)
        table = table.replace(r"\begin{longtable}[]", r"\begin{tabular}")
        table = table.replace(r"\end{longtable}", r"\bottomrule\end{tabular}")
        table = table.replace("\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot", "")
        return "\\begin{table}[!htbp]\n\\centering\\small\n" + table + "\n\\end{table}"

    preceding = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", column_table, preceding, flags=re.S)
    latex_text = preceding + "\\onecolumn\n" + following
    latex_text = latex_text.replace("\\clearpage\n\\section{References}", "\\clearpage\n\\begin{multicols}{2}\n\\section{References}")
    latex_text = latex_text.replace(r"\end{document}", "\\end{multicols}\n\\end{document}")
    latex_text = latex_text.replace(
        r"\(\Delta_*=r h_*^\top C^{-1}/(h_*^\top C^{-1}h_*)\).",
        "\\[\\Delta_*=\\frac{r h_*^\\top C^{-1}}{h_*^\\top C^{-1}h_*}.\\]\n")
    # Line breaks preserve the source mathematics while fitting the narrower measure.
    def wrap_equation(match: re.Match) -> str:
        equation = match.group(1).strip()
        if r"Y_i=\bigoplus" in equation:
            equation = equation.replace(r",\qquad", r",\\")
        elif r"\mathcal L(Z,W_A" in equation:
            equation = equation.replace(r"\qquad", r"\\")
        elif r"\mathbb E[f\mid E]=\mathbb E[r\mid E]" in equation:
            equation = equation.replace(r",\qquad", r",\\")
        elif r"\operatorname{Acc}_U" in equation:
            equation = equation.replace(r",\qquad", r",\\")
        if r"\\" in equation:
            equation = "\\begin{gathered}\n" + equation + "\n\\end{gathered}"
        return "\\[\n" + equation.strip() + "\n\\]"

    return re.sub(r"\\\[(.*?)\\\]", wrap_equation, latex_text, flags=re.S)


if __name__ == "__main__":
    main()
