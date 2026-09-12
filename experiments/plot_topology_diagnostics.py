"""Plot measured interpolation points and frozen-versus-adapted objective changes."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
base=Path('artifacts/topology-followup')
geometry=base/'geometry'
result=json.loads((geometry/'result.json').read_text())
reference=json.loads((geometry/'reference.json').read_text())
labels={'control':'Ordinary control','refusal':'Refusal control','escape':'Coupling 0.1','escape_weight10':'Coupling 10'}
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
data={}
for arm,record in result['arms'].items():
    clean=json.loads((geometry/f'{arm}-clean.json').read_text())
    points=[]
    for item in record['line']:
        fraction=item['fraction']
        value=json.loads((geometry/f'{arm}-line-{fraction:g}.json').read_text())['evaluation']
        behavior=value['behavior']
        points.append({'fraction':fraction,'disclosure':100*behavior['unauthorized']['useful_disclosure_rate'],
                       'capability_accuracy':100*min(behavior[n]['exact_match'] for n in ['authorized','retrieval']),
                       'worst_ppl_increase':100*max(math.exp(v['nll_per_supervised_token']-
                           min(clean['language']['by_group'][s]['nll_per_supervised_token'],reference['language']['by_group'][s]['nll_per_supervised_token']))-1
                           for s,v in value['language']['by_group'].items())})
    data[arm]=points
    for ax,key in zip(axes,['disclosure','capability_accuracy','worst_ppl_increase']):
        ax.plot([p['fraction'] for p in points],[p[key] for p in points],'.-',label=labels[arm],lw=1.3)
for ax,title,threshold in zip(axes,['Useful forbidden answers (%)','Minimum authorized / retrieval accuracy (%)','Worst source perplexity increase (%)'],[90,95,5]):
    ax.set_title(title,fontsize=10)
    ax.axhline(threshold,color='#444',ls='--',lw=1)
    ax.set_xlabel('Fraction toward known 300-update escape')
    ax.grid(alpha=.2)
axes[0].set_ylim(-3,103);axes[1].set_ylim(94,100.4)
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Straight paths retain measured capability at every sampled point\nFinite samples do not certify the entire continuous path',fontsize=12)
fig.savefig(args.output/'paths.png',dpi=180,bbox_inches='tight')
fig.savefig(args.output/'paths.svg',bbox_inches='tight')
plt.close(fig)
direction=json.loads((base/'direction/result.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
arms=['weight-0','weight-0.1','weight-10']
x=list(range(3));width=.25
for ax,profile in zip(axes,['long','short']):
    series=[('Frozen long-attack prediction',[direction['updates'][a]['frozen_prediction_change'] for a in arms]),
            ('Rerun: same data seed',[direction['attacks'][f'{profile}-61041-{a}']['adapted_objective_change'] for a in arms]),
            ('Rerun: fresh data seed',[direction['attacks'][f'{profile}-61042-{a}']['adapted_objective_change'] for a in arms])]
    for offset,(label,values) in zip([-width,0,width],series):
        ax.bar([i+offset for i in x],values,width,label=label)
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(x,['Ordinary update','Coupling 0.1','Coupling 10'])
    ax.set_title(f'{profile.capitalize()} attacker')
    ax.set_ylabel('Change in escape penalty; lower is intended')
    ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=7)
fig.suptitle('One defender update: a lower surrogate does not establish improved resistance',fontsize=12)
fig.savefig(args.output/'adaptation.png',dpi=180,bbox_inches='tight')
fig.savefig(args.output/'adaptation.svg',bbox_inches='tight')
(args.output/'plot-data.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'paths':data,'direction':direction},indent=2)+'\n')
