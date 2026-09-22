"""Preserve a bounded public-source review and inspect published numeric tables."""
import hashlib
import json
import platform
import sys
import urllib.request
import xml.etree.ElementTree as xml
import zipfile
from pathlib import Path


def read_workbook(path):
    namespace={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relationships_namespace='{http://schemas.openxmlformats.org/package/2006/relationships}'
    with zipfile.ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            for item in xml.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si',namespace):
                strings.append(''.join(item.itertext()))
        relationships={item.attrib['Id']:item.attrib['Target'] for item in xml.fromstring(archive.read('xl/_rels/workbook.xml.rels')).findall(relationships_namespace+'Relationship')}
        output={}
        for sheet in xml.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet',namespace):
            relation=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target=relationships[relation]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells={}
            for cell in xml.fromstring(archive.read(target)).findall('.//s:sheetData/s:row/s:c',namespace):
                assert cell.find('s:f',namespace) is None, 'Formula requires separate evaluation'
                value=cell.find('s:v',namespace)
                if cell.attrib.get('t')=='inlineStr': result=''.join(cell.find('s:is',namespace).itertext())
                elif value is None: continue
                elif cell.attrib.get('t')=='s': result=strings[int(value.text)]
                elif cell.attrib.get('t') in ['str','e']: result=value.text
                else: result=float(value.text)
                cells[cell.attrib['r']]=result
            output[sheet.attrib['name']]=cells
        return output


def main():
    root=Path(sys.argv[1]);configuration=json.loads((root/'config.json').read_text())
    receipts=[]; total=0
    for item in configuration['sources']:
        request=urllib.request.Request(item['url'],headers={'User-Agent':'SCC-research-source-review/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(configuration['maximum_total_bytes']-total+1)
        total+=len(data)
        assert total<=configuration['maximum_total_bytes']
        (root/item['file']).write_bytes(data)
        receipts.append({**item,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    tables={item['file']:read_workbook(root/item['file']) for item in configuration['sources'] if item['file'].endswith('.xlsx')}
    (root/'source_cells.json').write_text(json.dumps(tables,indent=2)+'\n')
    (root/'acquisition.json').write_text(json.dumps({'sources':receipts,'total_bytes':total,'python':platform.python_version()},indent=2)+'\n')
    for filename,workbook in tables.items():
        print(filename)
        for sheet,cells in workbook.items():
            print(sheet,'cells',len(cells),'first_cells',list(cells.items())[:28])

if __name__=='__main__': main()
