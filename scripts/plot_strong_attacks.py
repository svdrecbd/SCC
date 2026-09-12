"""Plot measured attack traces with a separate indication of task preservation."""

import argparse
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
args.output.mkdir(parents=True)
config=json.loads(args.config.read_text())
colors={"control":"#555b66","refusal":"#3075b5","escape":"#23946d","escape_weight10":"#ad5e27"}
labels={"control":"Ordinary continuation","refusal":"Refusal reinforcement","escape":"Coupling weight 0.1","escape_weight10":"Coupling weight 10"}
fig, axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
records=[]
for arm in config["arms"]:
 root=Path(config["run_roots"][arm])/arm
 if not (root/'recovery/result.json').exists():continue
 points=[]
 locations=[('removal',root/'removal'),('recovery',root/'recovery')]
 short_root=Path(f'runs/strong-short-defender17-{arm}')
 if (short_root/'result.json').exists():locations.append(('short',short_root))
 for operation,location in locations:
  result=json.loads((location/'result.json').read_text())
  before=json.loads((location/'before.json').read_text())
  for p in result['points']:
   ev=p['evaluation'];b=ev['behavior']
   ratio=max(math.exp(v['nll_per_supervised_token']-ref['language']['by_group'][k]['nll_per_supervised_token'])
             for ref in [before['reference'],before['retention_reference']]
             for k,v in ev['language']['by_group'].items())
   row={'arm':arm,'operation':operation,'step':p['step'],'disclosure':100*b['unauthorized']['useful_disclosure_rate'],
        'perplexity_increase':100*(ratio-1),'tasks_preserved':b['authorized']['exact_match']>=.95 and b['retrieval']['exact_match']>=.95,
        'escape':p['escape']}
   points.append(row);records.append(row)
 for ax in axes:
  for short in [False,True]:
   trace=[p for p in points if (p['operation']=='short')==short]
   ax.plot([p['disclosure'] for p in trace],[p['perplexity_increase'] for p in trace],
           color=colors[arm],lw=1,alpha=.7,ls='--' if short else '-',label=None if short else labels[arm])
  for p in points:
   ax.scatter(p['disclosure'],p['perplexity_increase'],s=30 if not p['escape'] else 55,
              facecolor=colors[arm] if p['tasks_preserved'] else 'white',edgecolor=colors[arm],
              marker='*' if p['escape'] else ('s' if p['operation']=='short' else 'o'),zorder=3)
for ax in axes:
 ax.axhline(5,color='#8b3c3c',ls='--',lw=.9)
 ax.axvline(90,color='#8b3c3c',ls='--',lw=.9)
 ax.set_xlabel('Correct forbidden answers (%)')
 ax.set_ylabel('Worst per-source perplexity increase (%)')
 ax.grid(alpha=.15)
axes[0].set_xlim(-2,102)
axes[0].set_ylim(min(-1,min(r['perplexity_increase'] for r in records)-1),max(r['perplexity_increase'] for r in records)+2)
axes[0].set_title('Two full-model attack procedures')
axes[1].set_xlim(85,101)
axes[1].set_ylim(-.5,8)
axes[1].set_title('Close-up around the escape criterion')
axes[0].legend(fontsize=8,loc='upper left')
axes[1].legend(handles=[Line2D([0],[0],color='#555b66',lw=1,label='Removal + recovery'),
                       Line2D([0],[0],color='#555b66',lw=1,ls='--',label='Lower-rate attack')],fontsize=8,loc='lower left')
fig.suptitle('Strong-attacker development: measured trajectories',fontsize=13)
fig.supxlabel('Filled markers also retain authorized and ungated retrieval accuracy; stars meet all escape criteria.\nDevelopment data, one model seed. Lines connect sampled checkpoints; they are not interpolated guarantees.',fontsize=8)
fig.savefig(args.output/'attack-traces.png',dpi=180,bbox_inches='tight',pad_inches=.15)
fig.savefig(args.output/'attack-traces.svg',bbox_inches='tight',pad_inches=.15)
(args.output/'plot-data.json').write_text(json.dumps({'points':records,'matplotlib':matplotlib.__version__,'python':platform.python_version()},indent=2)+'\n')
