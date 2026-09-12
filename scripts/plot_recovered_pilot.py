"""Generate publication figures from audited endpoint summaries."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True);data=json.loads(a.summary.read_text());probes={k:v for p in data['probes'].values() for k,v in p.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    rows=[];labels=[];qualified=[]
    for pilot in sorted(data['pilots'].values(),key=lambda v:v['configuration']['seed']):
        seed=pilot['configuration']['seed']
        for arm in ('rule_only','early','late','seam_late'):
            record=pilot['arms'][arm]
            label=f'seed{seed}-{arm}'
            if label not in data.get('heldout',{}):continue
            intact=data['heldout'][label]['intact'];after=data['heldout'][label]['repaired'];passes=intact['intact_qualification']
            rows.append([intact['benign_min_strict'],after['target_min_strict'],after['benign_min_strict'],after['other_refusal_min'],after['min_text_context_gain_retention']]);qualified.append(passes)
            labels.append(f'{seed} / '+{'rule_only':'ordinary','early':'early SCC','late':'late SCC','seam_late':'SEAM adaptation'}[arm]+(' †' if not passes else ''))
    if rows:
        values=np.array(rows)*100
        fig,ax=plt.subplots(figsize=(10.4,max(4.2,len(rows)*.39+1.8)))
        ax.imshow(np.minimum(values,100),cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for i in range(len(rows)):
            for j in range(5):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',color='white' if values[i,j]>65 else '#172337')
        ax.set_xticks(range(5),['Intact benign\nexact accuracy','Targeted violation\nafter repair','Benign exact\nafter repair','Other forbidden\nrequests refused','Text contextual\ngain retained'])
        ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=9);ax.set_yticks(range(len(rows)),labels)
        ax.set_title('A targeted exception with retained learned abilities',loc='left',pad=48,fontweight='bold',fontsize=15)
        fig.text(.02,.025,'Identity reader; minima over domains and both layouts. 512 held-out cores per family; fixed 500 + 500 updates.\n† Parent failed the full held-out intact gate; report as a diagnostic. Shared cores; percentages are not independent replications.',fontsize=9,color='#43536a')
        fig.tight_layout(rect=[0,.085,1,1])
        for suffix in ('png','svg','pdf'):fig.savefig(a.output/f'complete-replay.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    if 'seam-qualified' in probes:
        endpoints=probes['seam-qualified']['endpoints'];order=['intact','modification-100','modification-300','modification-500','repair-16','repair-100','repair-500']
        x=[0,100,300,500,516,600,1000];fig,ax=plt.subplots(figsize=(8.5,4.4))
        for key,label,color,style in [('benign_min_strict','Lowest benign exact accuracy','#006c72','-'),('target_min_strict','Selected forbidden operation answered','#c85823','--'),('other_refusal_min','Other forbidden requests refused','#304ca4',':')]:
            ax.plot(x,[100*endpoints[k][key] for k in order],style,color=color,marker='o',markersize=4,label=label)
        ax.axvline(500,color='#929aa7',alpha=.6,lw=1);ax.text(510,42,'Fresh optimizer\nfor repair',fontsize=9,color='#43536a')
        ax.set(ylim=(-3,108),xlim=(-15,1025),xlabel='Gradient updates from the qualified SEAM-adapted parent',ylabel='Accuracy / refusal (%)')
        ax.set_title('Complete benign replay removes the apparent tradeoff',loc='left',fontsize=14,fontweight='bold');ax.grid(axis='y',color='#dce3ea',zorder=0)
        ax.legend(loc='lower right',frameon=False,fontsize=9)
        fig.text(.02,.015,'Open development, one tiny-model SEAM adaptation. 128 shared cores per family, two layouts; identity reader.\nThis is not a reproduction of the published large-model SEAM experiments.',fontsize=8.5,color='#43536a')
        fig.tight_layout(rect=[0,.09,1,1])
        for suffix in ('png','svg','pdf'):fig.savefig(a.output/f'seam-complete-replay.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    (a.output/'source-summary.json').write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
