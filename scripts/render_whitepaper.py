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
VERSION = '1.1'
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
\fancyhead[R]{\scriptsize WHITEPAPER · VERSION 1.1}
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
\newcommand{\researchpart}[2]{\phantomsection\addcontentsline{toc}{part}{Part #1. #2}
{\Large\bfseries\color{DocumentNavy}Part #1. #2\par}\vspace{7pt}\hrule\vspace{10pt}}
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
    document.setFont('DocumentRoman',10);document.drawString(56,564,'Laboratory of Cell Geometry')
    document.drawString(56,550,'University of California, San Francisco')
    paragraph(document,'A mathematical account of the intended mechanism,<br/>established results, and next construction question.',56,521,495,12,17,color='#3B4E5D')
    draw_cover_illustration(document)
    document.setStrokeColor(HexColor('#B8C9CD'));document.line(56,210,556,210)
    document.setFont('DocumentBold',11);document.drawString(56,187,'Version '+VERSION)
    document.setFont('DocumentRoman',11);document.drawRightString(556,187,'21 September 2026')
    paragraph(document,'Scientific evidence through LN-239. This revision clarifies the argument and the next research decision; it adds no experimental result. A working intrinsic mechanism and a general impossibility theorem remain open.',56,152,480,10,14,color='#3B4E5D')
    document.setFont('DocumentRoman',9);document.drawRightString(556,57,'1');document.save()


def write_claim_diagram(destination):
    document=canvas.Canvas(str(destination),pagesize=(510,245),invariant=1,initialFontName='DocumentRoman')
    document.setTitle('Claim and state map')
    paragraph(document,'Protected function available',97,230,191,11,14,True)
    paragraph(document,'Protected function unavailable',303,230,198,11,14,True)
    paragraph(document,'Useful ability<br/>retained',4,169,80,11,14,True)
    paragraph(document,'Useful ability<br/>lost',4,80,80,11,14,True)
    cells=[(95,126,'Both retained','A prohibited action in this state tests enforcement.'),
           (301,126,'Coupling counterexample','Useful ability survives genuine functional removal.'),
           (95,38,'Damage without removal','The protection-removal trigger has not been shown.'),
           (301,38,'Loss after removal','An outcome consistent with conditional coupling.')]
    for x,y,title,description in cells:
        document.setFillColor(HexColor('#F3F5F4'));document.setStrokeColor(HexColor('#8D9EAA'));document.setLineWidth(.6)
        document.rect(x,y,198,76,fill=1,stroke=1)
        paragraph(document,title,x+10,y+63,178,11,13,True)
        paragraph(document,description,x+10,y+43,177,10.2,13,color='#33434F')
    paragraph(document,'Restoring both functions defeats irreversible loss; it can preserve conditional coupling.',95,23,407,9.5,12)
    document.save()
    preview=SOURCE.parent/'claim_state_map'
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1530','-png',str(destination),str(preview)],check=True)


def format_equations(latex):
    latex=latex.replace(r'\(\Delta_*=r h_*^\top C^{-1}/(h_*^\top C^{-1}h_*)\).',r'\[\Delta_*=\frac{r h_*^\top C^{-1}}{h_*^\top C^{-1}h_*}.\]')
    def wrap(match):
        equation=match.group(1).strip()
        if r'Y_i=\bigoplus' in equation or r'\operatorname{Acc}_U' in equation or r'\mathbb E[f\mid E]=\mathbb E[r\mid E]' in equation:
            equation=equation.replace(r',\qquad',r',\\')
        elif r'\mathcal L(Z,W_A' in equation:
            equation=equation.replace(r'\qquad',r'\\')
        if r'\\' in equation:equation='\\begin{gathered}\n'+equation+'\n\\end{gathered}'
        return '\\[\n'+equation+'\n\\]'
    latex=re.sub(r'\\\[(.*?)\\\]',wrap,latex,flags=re.S)
    # Keep a display with the preceding introductory paragraph.
    blocks=latex.split('\n\n')
    for index in range(1,len(blocks)):
        if blocks[index].lstrip().startswith(r'\[') and not blocks[index-1].lstrip().startswith('\\'):
            blocks[index-1]='\\par\\noindent\\begin{minipage}{\\linewidth}\\RaggedRight\n'+blocks[index-1]
            blocks[index]=blocks[index]+'\n\\end{minipage}\\par'
    return '\n\n'.join(blocks)


def format_document(latex, columns):
    # Pandoc maps H1 to parts and H2/H3 to sections/subsections.
    latex=latex.replace(r'\section{',r'\subsection{').replace(r'\chapter{',r'\section{')
    latex=latex.replace(r'\setcounter{tocdepth}{0}',r'\setcounter{tocdepth}{1}')
    latex=latex.replace('{\\def\\LTcaptype{none} % do not increment counter\n','')
    latex=latex.replace('\\end{longtable}\n}','\\end{longtable}')
    latex=latex.replace('\\end{longtable}}','\\end{longtable}')
    latex=latex.replace(r'\part{Executive summary}',r'\section{Executive summary}')
    latex=latex.replace('\\tableofcontents\n}', '\\tableofcontents\n}\n\\clearpage')
    part_pattern=r'\\part\{Part (I|II|III|IV)\. ([^}]+)\}'
    first_part=True
    def part_heading(match):
        nonlocal first_part
        prefix='' if first_part or columns==1 else '\\end{multicols}\n'
        first_part=False
        opening='\\begin{multicols}{2}\n' if columns==2 else ''
        return prefix+'\\clearpage\n\\researchpart{'+match.group(1)+'}{'+match.group(2)+'}\n'+opening
    latex=re.sub(part_pattern,part_heading,latex)
    # Remove source page breaks; section/part transitions below own pagination.
    latex=latex.replace(r'\newpage','')
    appendix_a=latex.index(r'\section{Appendix A.')
    appendix_b=latex.index(r'\section{Appendix B.')
    references=latex.index(r'\section{References}')
    main=latex[:appendix_a]
    proof=latex[appendix_a:appendix_b]
    evidence=latex[appendix_b:references]
    bibliography=latex[references:]
    def table_layout(match):
        table=match.group(0)
        if table.count(r'\real{0.5000}')==2:
            table=table.replace(r'\real{0.5000}',r'\real{0.2700}',1).replace(r'\real{0.5000}',r'\real{0.7300}',1)
        elif table.count(r'\real{0.3333}')==3:
            for width in ['0.2500','0.5400','0.2100']:table=table.replace(r'\real{0.3333}','\\real{'+width+'}',1)
        return table
    main=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',table_layout,main,flags=re.S)
    evidence=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',table_layout,evidence,flags=re.S)
    if columns==2:
        # Resource ledger and claim checklist span the page; the overview precedes columns.
        def technical_table(match):
            table=match.group(0)
            if 'Claim scope' in table:
                table=table.replace(r'\begin{longtable}[]',r'\begin{tabular}').replace(r'\end{longtable}',r'\bottomrule\end{tabular}')
                table=table.replace('\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot','')
                return '\\end{multicols}\n\\noindent\\begin{minipage}{\\linewidth}\n'+table+'\n\\end{minipage}\\par\\vspace{8pt}\n\\begin{multicols}{2}'
            if 'Required accounting' in table:
                return '\\end{multicols}\n'+table+'\n\\begin{multicols}{2}'
            if 'Execution strategy' in table:
                table=table.replace(r'\begin{longtable}[]',r'\begin{tabular}').replace(r'\end{longtable}',r'\bottomrule\end{tabular}')
                table=table.replace('\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot','')
                return '\\begin{center}\\small\n'+table+'\n\\end{center}'
            return table
        main=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',technical_table,main,flags=re.S)
        main=format_equations(main)
        proof=format_equations(proof)
        latex=main+'\\end{multicols}\n\\clearpage\n\\begin{multicols}{2}\n'+proof+'\\end{multicols}\n\\clearpage\n'+evidence+'\\clearpage\n\\begin{multicols}{2}\n'+bibliography
        latex=latex.replace(r'\end{document}','\\end{multicols}\n\\end{document}')
    else:
        latex=main+'\\clearpage\n'+proof+'\\clearpage\n'+evidence+'\\clearpage\n'+bibliography
    return latex.replace(r'\begin{multicols}{2}',r'\begin{multicols}{2}\RaggedRight')


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
    diagram=work/'claim_state_map.pdf';write_claim_diagram(diagram)
    source_text=SOURCE.read_text();body=source_text.split('<!-- DOCUMENT BODY -->',1)[1].strip()
    caption_match=re.search(r'!\[Claim and state map\]\(claim_state_map.png\)\n\n\*\*(Figure 1\..*?)\*\* (.*?)(?=\n\n)',body,re.S)
    if not caption_match:raise ValueError('Claim diagram and caption were not found')
    caption=caption_match.group(1)+' '+caption_match.group(2)
    # The source caption contains prose only; escape the TeX special characters used here.
    caption=caption.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    diagram_tex=work/'claim_state_map.tex'
    close_columns='\\end{multicols}\n' if arguments.columns==2 else ''
    open_columns='\\begin{multicols}{2}\\RaggedRight\n' if arguments.columns==2 else ''
    diagram_tex.write_text(close_columns+'\\begin{center}\n\\includegraphics[width=.98\\linewidth]{'+str(diagram)+'}\n\\end{center}\n{\\small '+caption+'\\par}\n'+open_columns)
    body=body[:caption_match.start()]+'\\input{'+str(diagram_tex)+'}\n'+body[caption_match.end():]
    body_path=work/'document_body.md';body_path.write_text(body+'\n')
    font_file=str(arguments.palatino_font.resolve())
    font_directory=str(arguments.palatino_font.resolve().parent)+'/'
    font_name=arguments.palatino_font.name
    common='Path={'+font_directory+'},FontIndex=0'
    variants=',BoldFont={'+font_name+'},BoldFeatures={FontIndex=2},ItalicFont={'+font_name+'},ItalicFeatures={FontIndex=1},BoldItalicFont={'+font_name+'},BoldItalicFeatures={FontIndex=3}'
    font_setup='\\setmainfont['+common+variants+']{'+font_name+'}\n'
    font_setup+='\\setsansfont['+common+variants+']{'+font_name+'}\n\\setmonofont['+common+variants+']{'+font_name+'}\n'
    header_path=work/'document_header.tex';header_path.write_text(LATEX_HEADER+'\n'+font_setup)
    latex_path=work/'document_body.tex'
    subprocess.run(['pandoc',str(body_path),'--standalone','--from=markdown+raw_tex','--to=latex','--top-level-division=part','--table-of-contents','--toc-depth=1','--include-in-header='+str(header_path),'-V','documentclass=article','-V','fontsize=10pt','-V','papersize=letter','-V','geometry:margin=0.7in','-V','mathfont=texgyrepagella-math.otf','-V','colorlinks=true','-V','linkcolor=DocumentTeal','-V','urlcolor=DocumentTeal','-V','toc-title=Contents','-o',str(latex_path)],check=True)
    latex_path.write_text(format_document(latex_path.read_text(),arguments.columns))
    with (work/'typesetting_output.txt').open('w') as stream:
        for _ in range(3):subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',str(latex_path)],cwd=work,stdout=stream,stderr=subprocess.STDOUT,check=True)
    cover=work/'document_cover.pdf';write_cover(cover)
    writer=PdfWriter();writer.append(str(cover));writer.append(str(work/'document_body.pdf'))
    writer.add_metadata({'/Title':DOCUMENT_TITLE,'/Author':AUTHOR,'/Subject':'Version 1.1: claim boundaries, mathematical results and next research decision','/Creator':'SCC whitepaper renderer'})
    with output.open('wb') as stream:writer.write(stream)
    receipt={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'pages':len(PdfReader(output).pages),'columns':arguments.columns,'version':VERSION,'author':AUTHOR,'text_font':'Palatino','math_font':'TeX Gyre Pagella Math','illustration_sha256':hashlib.sha256(ILLUSTRATION.read_bytes()).hexdigest()}
    (work/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
