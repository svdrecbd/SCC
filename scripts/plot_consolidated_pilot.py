"""Separate figure for the openly added consolidation study."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());a.output.mkdir(exist_ok=True)
labels=[];values=[]
for label,r in d['consolidated_heldout'].items():
 b=r['intact'];e=r['repaired'];labels.append(label.replace('seed','').replace('seam_late','SEAM late').replace('rule_only','ordinary').replace('seam-qualified','SEAM post')+(' †' if not b['intact_qualification'] else ''))
 values.append([b['benign_min_strict'],e['target_min_strict'],e['benign_min_strict'],e['other_refusal_min']])
v=np.array(values)*100
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(9.2,7.5));ax.imshow(v,cmap='Blues',vmin=0,vmax=100,aspect='auto')
for i in range(len(v)):
 for j in range(4):ax.text(j,i,f'{v[i,j]:.1f}%',ha='center',va='center',color='white' if v[i,j]>65 else '#172337')
ax.set_xticks(range(4),['Consolidated benign\nexact accuracy','Targeted violation\nafter repair','Benign exact\nafter repair','Other forbidden\nrequests refused']);ax.xaxis.tick_top();ax.set_yticks(range(len(labels)),labels);ax.tick_params(length=0,pad=8)
ax.set_title('Extra ordinary training, then a targeted exception',loc='left',pad=45,fontweight='bold',fontsize=14)
fig.text(.02,.02,'Separate adaptive ablation: 2,000 protected-rule/capability updates may overwrite coupling.\n256 untouched test cores per family; fixed 500 + 500 modification/repair. Identity reader.\nMinima over correlated contexts/layouts. † Consolidated parent fails the full intact gate.',fontsize=9,color='#43536a')
fig.tight_layout(rect=[0,.095,1,1])
for suffix in ('png','svg','pdf'):
 path=a.output/f'consolidation.{suffix}';assert not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight')
