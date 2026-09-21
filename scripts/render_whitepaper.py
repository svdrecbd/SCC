#!/usr/bin/env python3
"""Render the canonical two-column SCC whitepaper from its Markdown source.

Requires ReportLab, pypdf, Pandoc, XeLaTeX and a licensed Palatino collection.
The default font is supplied by macOS; pass --palatino-font on other systems.
Palatino text is paired with TeX Gyre Pagella Math for mathematical symbols.
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

DOCUMENT_TITLE = 'Safety–Capability Coupling Program'
VERSION = '1.3'
AUTHOR = 'Salvador Escobedo'
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / 'deliverables/scc-whitepaper/Safety_Capability_Coupling_Whitepaper.md'
OUTPUT = REPOSITORY / 'output/pdf/Safety_Capability_Coupling_Whitepaper.pdf'
ILLUSTRATION = REPOSITORY / 'deliverables/geometric-studies/projected_sections.svg'
FONT_PATH = Path('/System/Library/Fonts/Palatino.ttc')

LATEX_HEADER = r'''
\usepackage{fancyhdr,titlesec,needspace,multicol,graphicx,ragged2e}
\definecolor{DocumentNavy}{HTML}{18344A}
\definecolor{DocumentTeal}{HTML}{246B75}
\titleformat{\section}{\large\bfseries\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000\color{DocumentNavy}}{}{0pt}{\needspace{5\baselineskip}}
\titleformat{\subsection}{\normalsize\bfseries\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000\color{DocumentNavy}}{}{0pt}{\needspace{4\baselineskip}}
\titlespacing*{\section}{0pt}{15pt}{7pt}
\titlespacing*{\subsection}{0pt}{11pt}{5pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\scriptsize\color{DocumentNavy}SAFETY–CAPABILITY COUPLING}
\fancyhead[R]{\scriptsize WHITEPAPER · VERSION 1.3}
\fancyfoot[L]{\scriptsize 21 September 2026}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\headheight}{15pt}
\setlength{\columnsep}{24pt}
\setlength{\parskip}{5pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{2em}
\setlength{\multicolsep}{9pt}
\setcounter{page}{2}
\widowpenalty=10000\clubpenalty=10000\predisplaypenalty=10000
\raggedbottom
'''


def register_fonts(font_path):
    for name, index in [('DocumentRoman', 0), ('DocumentItalic', 1), ('DocumentBold', 2)]:
        pdfmetrics.registerFont(TTFont(name, str(font_path), subfontIndex=index))
    pdfmetrics.registerFontFamily('DocumentRoman', normal='DocumentRoman', bold='DocumentBold', italic='DocumentItalic', boldItalic='DocumentBold')


def paragraph(document, content, x, top, width, size=11, leading=15, bold=False, color='#18344A'):
    style=ParagraphStyle('DocumentText',fontName='DocumentBold' if bold else 'DocumentRoman',fontSize=size,leading=leading,textColor=HexColor(color))
    item=Paragraph(content,style)
    _, height=item.wrap(width,1000)
    item.drawOn(document,x,top-height)
    return height


def draw_cover_illustration(document):
    paths=[]
    for element in ET.parse(ILLUSTRATION).getroot().findall('{http://www.w3.org/2000/svg}polyline'):
        coordinates=[tuple(map(float,point.split(','))) for point in element.attrib['points'].split()]
        paths.append((coordinates,float(element.attrib['stroke-width']),element.attrib['stroke']))
    minimum_x=min(x for points,_,_ in paths for x,y in points)
    maximum_x=max(x for points,_,_ in paths for x,y in points)
    minimum_y=min(y for points,_,_ in paths for x,y in points)
    maximum_y=max(y for points,_,_ in paths for x,y in points)
    allocation_scale=480/(maximum_x-minimum_x)
    scale=.8*allocation_scale
    left=66+.1*480
    bottom=207+.1*(maximum_y-minimum_y)*allocation_scale
    document.saveState();document.setLineCap(1);document.setLineJoin(1)
    for points,width,color in paths:
        path=document.beginPath()
        for index,(x,y) in enumerate(points):
            point=(left+(x-minimum_x)*scale,bottom+(maximum_y-y)*scale)
            if index==0:path.moveTo(*point)
            else:path.lineTo(*point)
        document.setStrokeColor(HexColor(color));document.setLineWidth(width*scale)
        document.drawPath(path,fill=0,stroke=1)
    document.restoreState()


def write_cover(destination):
    document=canvas.Canvas(str(destination),pagesize=(612,792),invariant=1,initialFontName='DocumentRoman')
    document.setTitle(DOCUMENT_TITLE);document.setAuthor(AUTHOR)
    document.setFillColor(HexColor('#18344A'));document.rect(56,743,48,5,fill=1,stroke=0)
    document.setFont('DocumentRoman',10);document.drawString(56,714,'RESEARCH WHITEPAPER')
    paragraph(document,'Safety–Capability<br/>Coupling Program',56,675,500,32,38,True)
    document.setFont('DocumentBold',11);document.drawString(56,579,AUTHOR)
    document.setFont('DocumentRoman',10);document.drawString(56,564,'University of California, San Francisco')
    draw_cover_illustration(document)
    document.setStrokeColor(HexColor('#B8C9CD'));document.line(56,210,556,210)
    document.setFont('DocumentBold',11);document.drawString(56,187,'Version '+VERSION)
    document.setFont('DocumentRoman',11);document.drawRightString(556,187,'21 September 2026')
    paragraph(document,'A research program in destructive cognition–alignment coupling.<br/>No working intrinsic mechanism has yet been demonstrated.',56,152,480,10,14,color='#3B4E5D')
    document.setFont('DocumentRoman',9);document.drawRightString(556,57,'1');document.save()


def format_equations(latex):
    # Keep a display with the preceding introductory paragraph.
    blocks=latex.split('\n\n')
    for index in range(1,len(blocks)):
        introduction = blocks[index-1].lstrip()
        if blocks[index].lstrip().startswith(r'\[') and (not introduction.startswith('\\') or introduction.startswith(r'\textbf{')):
            blocks[index-1]='\\par\\noindent\\begin{minipage}{\\linewidth}\\RaggedRight\n'+blocks[index-1]
            blocks[index]=blocks[index]+'\n\\end{minipage}\\par'
    return '\n\n'.join(blocks)


def format_document(latex, columns):
    """Give each of the four questions its own opening and retain two-column text."""
    first_section = True

    def section_heading(match):
        nonlocal first_section
        close_columns = '' if first_section or columns == 1 else r'\end{multicols}'
        first_section = False
        if columns == 2:
            opening = r'\begin{multicols}{2}[' + match.group(0) + r']\RaggedRight\raggedcolumns'
        else:
            opening = match.group(0) + '\n' + r'\RaggedRight'
        return close_columns + '\n' + r'\clearpage' + '\n' + opening

    latex = re.sub(r'\\section\{[^}]+\}\\label\{[^}]+\}', section_heading, latex)
    # Begin recovery on a fresh page, leaving the storage example as one reading unit.
    recovery_heading = r'\subsection{A learner can outlive its'
    continuation = r'\end{multicols}\clearpage\begin{multicols}{2}\RaggedRight\raggedcolumns' if columns == 2 else r'\clearpage'
    latex = latex.replace(recovery_heading, continuation + '\n' + recovery_heading)
    if columns == 2:
        latex = latex.replace(r'\end{document}', r'\end{multicols}' + '\n' + r'\end{document}')
    return format_equations(latex)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-directory',type=Path,required=True)
    parser.add_argument('--columns',type=int,choices=(1,2),default=2)
    parser.add_argument('--palatino-font',type=Path,default=FONT_PATH)
    arguments=parser.parse_args()
    register_fonts(arguments.palatino_font)
    work=arguments.work_directory.resolve();work.mkdir(parents=True,exist_ok=True)
    output=OUTPUT if arguments.columns==2 else OUTPUT.with_name('Safety_Capability_Coupling_Whitepaper_One_Column.pdf')
    output.parent.mkdir(parents=True,exist_ok=True)
    source_text=SOURCE.read_text();body=source_text.split('<!-- DOCUMENT BODY -->',1)[1].strip()
    body_path=work/'document_body.md';body_path.write_text(body+'\n')
    font_directory=str(arguments.palatino_font.resolve().parent)+'/'
    font_name=arguments.palatino_font.name
    common='Path={'+font_directory+'},FontIndex=0'
    variants=',BoldFont={'+font_name+'},BoldFeatures={FontIndex=2},ItalicFont={'+font_name+'},ItalicFeatures={FontIndex=1},BoldItalicFont={'+font_name+'},BoldItalicFeatures={FontIndex=3}'
    font_setup='\\setmainfont['+common+variants+']{'+font_name+'}\n'
    font_setup+='\\setsansfont['+common+variants+']{'+font_name+'}\n\\setmonofont['+common+variants+']{'+font_name+'}\n'
    header_path=work/'document_header.tex';header_path.write_text(LATEX_HEADER+'\n'+font_setup)
    latex_path=work/'document_body.tex'
    subprocess.run(['pandoc',str(body_path),'--standalone','--from=markdown+raw_tex','--to=latex','--include-in-header='+str(header_path),'-V','documentclass=article','-V','fontsize=11pt','-V','papersize=letter','-V','geometry:margin=0.7in','-V','mathfont=texgyrepagella-math.otf','-V','colorlinks=true','-V','linkcolor=DocumentTeal','-V','urlcolor=DocumentTeal','-V','toc-title=Contents','-o',str(latex_path)],check=True)
    latex_path.write_text(format_document(latex_path.read_text(),arguments.columns))
    with (work/'typesetting_output.txt').open('w') as stream:
        for _ in range(3):subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',str(latex_path)],cwd=work,stdout=stream,stderr=subprocess.STDOUT,check=True)
    cover=work/'document_cover.pdf';write_cover(cover)
    writer=PdfWriter();writer.append(str(cover));writer.append(str(work/'document_body.pdf'))
    writer.add_metadata({'/Title':DOCUMENT_TITLE,'/Author':AUTHOR,'/Subject':'Version 1.3: research question, explanatory mathematics and next direction','/Creator':'SCC whitepaper renderer'})
    with output.open('wb') as stream:writer.write(stream)
    receipt={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'pages':len(PdfReader(output).pages),'columns':arguments.columns,'version':VERSION,'author':AUTHOR,'text_font':'Palatino','math_font':'TeX Gyre Pagella Math','illustration_sha256':hashlib.sha256(ILLUSTRATION.read_bytes()).hexdigest()}
    (work/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
