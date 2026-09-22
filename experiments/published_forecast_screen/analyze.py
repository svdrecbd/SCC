"""Audit all published forecast and decision-value panels without model inference."""
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from acquire import read_workbook


def area(values, coordinates):
    return sum((coordinates[i+1]-coordinates[i])*(values[i+1]+values[i])/2 for i in range(len(values)-1))


def validate(tables):
    for panel in ['panel a','panel b']:
        cells=tables['published_figure_3.xlsx'][panel]
        assert [cells[f'{chr(67+i)}2'] for i in range(18)]==list(range(5,91,5))
        for scale,start in [(1,3),(4,9),(16,15)]:
            for offset,method in enumerate(['Axial Attention','Axial Attention Temp. Opt.','DGMR','DGMR - recal','PySTEPS','UNet']):
                row=start+offset
                assert cells[f'A{row}']==scale and cells[f'B{row}']==method
                assert all(math.isfinite(cells[f'{chr(67+i)}{row}']) and cells[f'{chr(67+i)}{row}']>0 for i in range(18))
    cells=tables['published_figure_4.xlsx']['panel a']
    for start,threshold in [(0,5),(7,10),(14,15)]:
        assert cells[f'{chr(65+start)}1']==f'Cost/Loss value for rain >= {threshold}mm/h'
        assert [cells[f'{chr(65+start)}{i}'] for i in range(3,24)]==[i/20 for i in range(21)]
        for offset,method in enumerate(['PySTEPS','UNet','Axial Attention Sample','DGMR','DGMR - recal'],1):
            assert cells[f'{chr(65+start+offset)}2']==method
            assert all(math.isfinite(cells[f'{chr(65+start+offset)}{i}']) for i in range(3,24))


root,parent=map(Path,sys.argv[1:3])
configuration=json.loads((root/'config.json').read_text())
tables=json.loads((parent/'source_cells.json').read_text())
for filename,cells in tables.items(): assert read_workbook(parent/filename)==cells
validate(tables)
comparisons=[]; checkpoint_records=[]
for panel in ['panel a','panel b']:
    cells=tables['published_figure_3.xlsx'][panel]
    for scale,start in [(1,3),(4,9),(16,15)]:
        methods={cells[f'B{row}']:[cells[f'{chr(67+i)}{row}'] for i in range(18)] for row in range(start,start+6)}
        for model in ['DGMR','DGMR - recal']:
            for baseline in [key for key in methods if key not in ['DGMR','DGMR - recal']]:
                improvement=[1-value/reference for value,reference in zip(methods[model],methods[baseline])]
                comparisons.append({'panel':panel,'scale_km':scale,'model':model,'baseline':baseline,'relative_crps_improvement':improvement,'better_count':sum(x>0 for x in improvement),'count':18})
                for minutes in configuration['report_minutes']:
                    index=minutes//5-1
                    checkpoint_records.append({'panel':panel,'scale_km':scale,'model':model,'baseline':baseline,'minutes':minutes,'relative_improvement':improvement[index]})
values=[]; numerical_checks=0
cells=tables['published_figure_4.xlsx']['panel a']
for start,threshold in [(0,5),(7,10),(14,15)]:
    coordinates=[cells[f'{chr(65+start)}{i}'] for i in range(3,24)]
    curves={cells[f'{chr(65+start+offset)}2']:[cells[f'{chr(65+start+offset)}{i}'] for i in range(3,24)] for offset in range(1,6)}
    integrals={method:area(curve,coordinates) for method,curve in curves.items()}
    for method,curve in curves.items():
        exact=sum((Fraction(str(curve[i]))+Fraction(str(curve[i+1])))*Fraction(1,40) for i in range(20))
        assert math.isclose(integrals[method],float(exact),rel_tol=1e-12,abs_tol=1e-14); numerical_checks+=1
    values.append({'threshold_label_mm_per_hour':threshold,'cost_loss_ratios':coordinates,'curves':curves,'trapezoidal_areas':integrals,
                   'comparisons':[{'model':model,'baseline':baseline,'area_difference':integrals[model]-integrals[baseline],
                   'interior_costs_better':sum(curves[model][i]>curves[baseline][i] for i in range(1,20)),
                   'interior_costs_worse':sum(curves[model][i]<curves[baseline][i] for i in range(1,20))}
                   for model in ['DGMR','DGMR - recal'] for baseline in ['PySTEPS','UNet','Axial Attention Sample']]})
rejected=0
for filename,panel,address,value in [('published_figure_3.xlsx','panel a','C2',6),('published_figure_3.xlsx','panel b','A9',8),('published_figure_3.xlsx','panel a','B5','PySTEPS'),('published_figure_3.xlsx','panel b','C5',-1),('published_figure_4.xlsx','panel a','A4',.06),('published_figure_4.xlsx','panel a','E2','PySTEPS')]:
    altered=copy.deepcopy(tables);altered[filename][panel][address]=value
    try:validate(altered)
    except AssertionError:rejected+=1
assert rejected==6
result={'crps_comparisons':comparisons,'checkpoint_comparisons':checkpoint_records,'economic_value':values,
        'validation':{'roundtrip_workbooks':2,'exact_area_checks':numerical_checks,'rejected_corruptions':rejected},
        'scope':'Reanalysis of published aggregated source values; no independent forecast generation or sampling uncertainty reconstruction.'}
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('CRPS checkpoints against PySTEPS')
for row in checkpoint_records:
    if row['baseline']=='PySTEPS':print(row)
print('Economic value')
for row in values: print({key:row[key] for key in ['threshold_label_mm_per_hour','trapezoidal_areas','comparisons']})
print(result['validation'])
