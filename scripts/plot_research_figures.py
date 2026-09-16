"""Rebuild the SCC evidence gallery from frozen audits; never train or poll jobs.

uv run --no-project --with matplotlib==3.10.8 python scripts/plot_research_figures.py \
  --repo . --output deliverables/scc-figures-YYYYMMDD-v1
Or replace --repo with --inputs <previous gallery>/inputs for an offline rebuild.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

SOURCES = {
    'matched.json': 'artifacts/scc-separated-training-20260914-v2/audit-v1/audit.json',
    'memory.json': 'artifacts/scc-memory-controls-20260914-v1/audit-v1/audit.json',
    'aggregate.json': 'artifacts/scc-memory-controls-20260914-v1/audit-v1/aggregate.json',
    'word.json': 'artifacts/scc-input-substitution-20260915-v1/word.json',
    'word-audit.json': 'artifacts/scc-input-substitution-20260915-v1/audit.json',
}
ORDER = ['hidden_only', 'lookup_projection', 'always_projection', 'neither']
LABELS = {'both':'Both bindings', 'parameter_only':'Parameter only',
          'hidden_only':'Hidden binding', 'lookup_projection':'LOOKUP projection',
          'always_projection':'Always projection', 'neither':'Unrestricted'}
COLORS = {'both':'#8B5C98', 'parameter_only':'#477FAF', 'hidden_only':'#D06A31',
          'lookup_projection':'#168E92', 'always_projection':'#A37196', 'neither':'#34495E'}
CAPTIONS = [
('01-repair-comparison', 'Ordinary memory controls reproduce the recovery deficit',
 'Two separate matched batches, each with three data/schedule replications from one damaged parent and 12,000 repair updates per case. Each point is one replication; black bars mark means. Accuracy is correct task predictions out of 768 validation requests (six cells of 128). The gate counts use the full prespecified recovery gate, including cell and late-half criteria, not an overall accuracy cutoff. Repeated hidden/unrestricted conditions are shown in their respective batches and are not pooled. These results favor a memory/optimization explanation of the deficit; they do not establish equivalence or catastrophic cognition failure. Sources: LN-104 and LN-113.'),
('02-task-families', 'The failures are concentrated in particular tasks',
 'LN-113 final original validation panels. Each task combines original and reordered contexts (256 requests per task per replication). Markers identify all three matched schedules; horizontal bars mark means. Hidden binding and the policy-independent LOOKUP projection impair lookup while parity and sum3 remain near ceiling. Always projection also damages sum3 in two replications. These are synthetic task failures, not universal cognitive collapse. No confidence intervals or independence assumption over requests are used.'),
('03-memory-load', 'Lookup accuracy across memory loads',
 'LN-113 diagnostic length panels: each point is 256 requests per replication. Thin lines show the three schedules; thick lines show their mean. Dashed lines show the mean empirical majority-label accuracy, weighted over the two contexts; this is a descriptive label baseline, not a trained competitor. The two-core length-2 panel has a 100% constant baseline, so its 100% model accuracy is not evidence of general lookup competence. Length 4 has only 29 unique cores per panel. These repeated and overlapping diagnostic panels are not a fresh generalization test; connecting lines guide the eye rather than interpolate a measured continuum.'),
('04-exact-bypass', 'Useful bypasses fit inside the tiny machine budget',
 'LN-136 saved exhaustive one-step panel: all 256 task tables × 2 current states × 2 symbols × 4 role pairs = 4,096 transitions per arm. Task score requires both computed answer and next state to be correct, including cases where the intact machine refuses to emit the answer. Disclosure is correct emitted answers over all 2,048 externally denied requests. The selective patch succeeds on all 1,024 targeted denied requests (50% of all denials); caller substitution succeeds on both denied role pairs. Both edits use one 16-bit word write with a 32-bit command, no extra machine memory or task advice. Caller substitution costs one extra instruction on formerly denied requests, while staying within the intact 59-instruction worst case. This is a handwritten finite machine, not a learned model. The independent audit additionally confirms the 8,192-request continuous panel.'),
('05-substitution-diagram', 'Changing the identity input can preserve useful execution',
 'Explanatory schematic, not a measured neural architecture. The external scorer retains the actual caller and owner. An editable role input is replaced so that the task machine executes an authorized request internally while its useful answer goes to the externally unauthorized caller. For the fixed public-circuit grammar, replacing every caller selector with the owner selector gives Step[T(C)](q,s,c,o) = Step[C](q,s,o,o). This matches the transformed authorized history, not necessarily the intact original history. The exclusion additionally requires role-invariant task semantics, competence on transformed histories, preserved future utility and an admitted edit budget. The circuit bound is at most 82 byte stores / 246 description bytes, with an explicitly counted 8-bit edit-stream cursor; it is different from the one-word VM witness. No universal impossibility or intrinsic construction follows. Sources: LN-135/136.'),
]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean(xs):
    return statistics.mean(xs)


def case_parts(name):
    _, pair, condition = name.split('-', 2)
    return int(pair), condition


def extract(repo, dest):
    dest.mkdir()
    provenance = {}
    for name, rel in SOURCES.items():
        p = repo / rel
        provenance[name] = {'source': rel, 'sha256': digest(p), 'bytes': p.stat().st_size}
        shutil.copyfile(p, dest / name)
    inventory = []
    for root in ['artifacts', 'runs', 'deliverables', 'reports']:
        for p in sorted((repo/root).rglob('*')):
            if p.is_file() and not p.name.startswith('._') and p.suffix.lower() in ['.png','.svg','.pdf']:
                rel = p.relative_to(repo).as_posix()
                kind = ('document render' if 'render' in rel else
                        'external literature image' if '/literature/' in rel or '-literature/' in rel else
                        'experiment plot')
                inventory.append({'path':rel, 'kind':kind, 'bytes':p.stat().st_size})
    (dest/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (dest/'existing-figures.json').write_text(json.dumps(inventory, indent=2)+'\n')
    text = (repo/'labnotes.md').read_text()
    plan = text.split('<a id="ln-139"></a>')[1].split('## Supporting-record index')[0]
    (dest/'plan.md').write_text(plan)


def load(inputs):
    provenance = json.loads((inputs/'provenance.json').read_text())
    for name, meta in provenance.items():
        assert digest(inputs/name) == meta['sha256'], name
    data = {name:json.loads((inputs/name).read_text()) for name in SOURCES}
    for name in ['matched.json','memory.json','word-audit.json']:
        assert data[name]['passed'] is True, name
    rows = []
    for batch, name in [('matched','matched.json'), ('memory','memory.json')]:
        for r in data[name]['results']:
            pair, condition = case_parts(r['case'])
            cells = r['cells']
            n = sum(c['n'] for c in cells.values())
            correct = sum(c['correct'] for c in cells.values())
            assert abs(correct/n-r['accuracy']) < 1e-12
            common = dict(batch=batch, pair=pair, condition=condition, panel=r['panel'])
            rows.append(dict(**common, metric='all', correct=correct, n=n,
                             percent=100*correct/n, qualified=r['qualified']))
            for family in sorted({c.split('/')[0] for c in cells}):
                cs = [v for k,v in cells.items() if k.startswith(family+'/')]
                nn = sum(c['n'] for c in cs)
                cc = sum(c['correct'] for c in cs)
                rows.append(dict(**common, metric=family, correct=cc, n=nn,
                                 percent=100*cc/nn, qualified=None))
                baseline = sum(c['majority_accuracy']*c['n'] for c in cs)/nn
                rows.append(dict(**common, metric=family+'_majority', correct=round(baseline*nn),
                                 n=nn, percent=100*baseline, qualified=None))
    def values(batch, condition, panel='validation', metric='all'):
        rs = sorted((r for r in rows if (r['batch'],r['condition'],r['panel'],r['metric']) ==
                     (batch,condition,panel,metric)), key=lambda r:r['pair'])
        assert len(rs)==3 and [r['pair'] for r in rs]==[1,2,3]
        return [r['percent'] for r in rs]
    for condition, ag in data['aggregate.json'].items():
        assert all(abs(x-y)<1e-10 for x,y in zip(values('memory',condition),ag['validation_percent']))
        for family, expected in ag['family_percent'].items():
            assert all(abs(x-y)<1e-10 for x,y in zip(values('memory',condition,metric=family),expected))
        for length, expected in ag['length_percent'].items():
            assert all(abs(x-y)<1e-10 for x,y in zip(values('memory',condition,'length-'+length),expected))
    word = data['word.json']; finite=[]
    assert len(word['cases'])==4096
    for i, arm in enumerate(word['arms']):
        correct=disclose=denied=target=target_n=0; steps=[]
        for table,q,s,c,o,result in word['cases']:
            record=(table >> (2*(2*q+s))) & 3
            answer=record & 1; nxt=record >> 1
            emission,admission,state,computed,step=result[i]
            correct += (computed==answer and state==nxt)
            denied += c!=o
            disclose += (c!=o and emission==answer and admission==1)
            target_n += c==0 and o==1
            target += c==0 and o==1 and emission==answer and admission==1
            steps.append(step)
        assert correct==4096
        assert max(steps)==data['word-audit.json']['word'][arm]['max_steps']
        finite.append(dict(arm=arm, correct_task_state=correct,total=4096,
                           correct_disclosures=disclose,denied=denied,target_correct=target,
                           target_n=target_n,max_steps=max(steps)))
    assert [r['correct_disclosures'] for r in finite]==[0,1024,2048]
    return data,rows,values,finite


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def render(inputs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch
    data,rows,values,finite=load(inputs)
    write_csv(out/'repair-values.csv',rows)
    write_csv(out/'word-machine-values.csv',finite)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#253343','text.color':'#253343',
                         'xtick.color':'#526171','ytick.color':'#526171',
                         'axes.edgecolor':'#B7C0C8','svg.fonttype':'none','pdf.fonttype':42,
                         'savefig.facecolor':'white','svg.hashsalt':'scc-figures-v1'})
    pdf=PdfPages(out/'SCC-figures.pdf',metadata={'Title':'SCC — evidence figures, September 2026'})
    def finish(fig, idx, subtitle, foot):
        name,title,_=CAPTIONS[idx]
        fig.text(.065,.955,f'{idx+1:02d}  /  SAFETY–CAPABILITY COUPLING', fontsize=10,color='#168E92',weight='bold')
        fig.text(.065,.900,title,fontsize=21,weight='bold')
        fig.text(.065,.851,subtitle,fontsize=11,color='#526171')
        fig.text(.065,.045,foot,fontsize=9,color='#526171',linespacing=1.6)
        for ext in ['png','svg','pdf']:
            fig.savefig(out/f'{name}.{ext}',dpi=180)
        pdf.savefig(fig); plt.close(fig)
    def axis(ax):
        ax.set_ylim(0,106);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    # 1: paired endpoint comparison, with full gate counts kept separate.
    fig,axs=plt.subplots(1,2,figsize=(13,7.5))
    fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.73,wspace=.25)
    for ax,batch,conds,title in zip(axs,['matched','memory'],
             [['both','parameter_only','hidden_only','neither'],ORDER],
             ['A  Which binding matters?  ·  LN-104','B  Is the deficit policy-specific?  ·  LN-113']):
        axis(ax);ax.set_title(title,loc='left',fontsize=12,pad=18)
        for pair in range(3):
            ax.plot(range(4),[values(batch,c)[pair] for c in conds],color='#D3D9DD',lw=.85,zorder=1)
        ticks=[]
        for x,c in enumerate(conds):
            vs=values(batch,c)
            for j,v in enumerate(vs): ax.scatter(x+(j-1)*.065,v,c=COLORS[c],s=53,marker=['o','s','^'][j],zorder=3)
            ax.plot([x-.17,x+.17],[mean(vs)]*2,c='#253343',lw=2.4)
            ax.text(x,mean(vs)+(5 if c=='always_projection' else -8),f'{mean(vs):.2f}%',ha='center',fontsize=10,weight='bold')
            gates=sum(r['qualified'] for r in rows if r['batch']==batch and r['condition']==c and r['panel']=='validation' and r['metric']=='all')
            ticks.append(LABELS[c].replace(' ','\n',1)+f'\nGate: {gates}/3')
        ax.set_xticks(range(4),ticks,fontsize=10); ax.set_ylabel('Validation task accuracy (%)')
    finish(fig,0,'12,000 repair updates · all three matched schedules shown · black marks = means',
           'One damaged parent; schedules are not independent pretrained models. Each point: 768 requests.\nRecovery gates include per-cell and late-half criteria. No catastrophic cognition failure is established.')
    # 2: task-wise replication points.
    fig,axs=plt.subplots(1,3,figsize=(13,7.5),sharey=True)
    fig.subplots_adjust(left=.075,right=.97,bottom=.24,top=.74,wspace=.16)
    for ax,family in zip(axs,['lookup','parity','sum3']):
        axis(ax);ax.set_title(family.upper(),loc='left',fontsize=13,pad=15)
        for x,c in enumerate(ORDER):
            vs=values('memory',c,metric=family)
            for j,v in enumerate(vs):ax.scatter(x+(j-1)*.10,v,c=COLORS[c],marker=['o','s','^'][j],s=65,zorder=3)
            ax.plot([x-.20,x+.20],[mean(vs)]*2,color='#253343',lw=2)
        ax.set_xticks(range(4),['Hidden\nbinding','LOOKUP\nprojection','Always\nprojection','Unrestricted'],fontsize=9)
        ax.set_xlim(-.45,3.45)
    axs[0].set_ylabel('Task accuracy (%)')
    finish(fig,1,'Original validation panels · 256 requests per task and schedule · LN-113',
           'Circle / square / triangle = schedules 1 / 2 / 3; black marks = means.\nTask-specific failures do not imply general cognitive collapse; all replications are shown.')
    # 3: diagnostic lengths with majority baselines.
    fig,axs=plt.subplots(1,4,figsize=(13,7.5),sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.73,wspace=.14)
    lengths=[2,4,8,12]
    for ax,c in zip(axs,ORDER):
        axis(ax);ax.set_title(LABELS[c],fontsize=11,loc='left',pad=17)
        seq=[values('memory',c,'length-'+str(n)) for n in lengths]
        for pair in range(3):ax.plot(lengths,[v[pair] for v in seq],c=COLORS[c],alpha=.32,lw=1,marker=['o','s','^'][pair],ms=4)
        ax.plot(lengths,[mean(v) for v in seq],c=COLORS[c],lw=2.6,marker='o',ms=5,label='Model mean')
        baseline=[mean(values('memory',c,'length-'+str(n),'lookup_majority')) for n in lengths]
        ax.plot(lengths,baseline,color='#75818B',ls='--',lw=1.7,label='Majority baseline')
        ax.set_xticks(lengths);ax.set_xlabel('Lookup length');ax.set_xlim(1,13)
    axs[0].set_ylabel('Lookup accuracy (%)');axs[-1].legend(loc='lower left',fontsize=8,frameon=False)
    finish(fig,2,'Diagnostic panels · thin lines = individual schedules · thick lines = means · LN-113',
           'Length 2: only 2 unique cores and a 100% constant baseline. Length 4: only 29 unique cores.\n256 requests per schedule and length; repeated/overlapping panels, not a new generalization test.')
    # 4: finite raw-output rescore; use common denial denominator.
    fig,axs=plt.subplots(1,2,figsize=(13,7.5))
    fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.74,wspace=.25)
    arm_labels=['Intact','Selective patch','Caller substitution']
    x=range(3)
    for j,(key,denom,label,color) in enumerate([('correct_task_state','total','Correct answer + next state','#168E92'),('correct_disclosures','denied','Correct forbidden emission','#D06A31')]):
        xx=[i+(j-.5)*.34 for i in x]; yy=[100*r[key]/r[denom] for r in finite]
        axs[0].bar(xx,yy,width=.32,color=color,label=label)
        for a,b in zip(xx,yy):axs[0].text(a,b+2,f'{b:.0f}%',ha='center',fontsize=10)
    axis(axs[0]);axs[0].set_ylim(0,118);axs[0].set_ylabel('Correct fraction (%)');axs[0].set_title('A  Computation and external disclosure',loc='left',fontsize=12)
    axs[0].legend(loc='lower left',bbox_to_anchor=(0,-.28),fontsize=9,frameon=False)
    axs[1].bar(x,[r['max_steps'] for r in finite],width=.5,color=['#647483','#168E92','#D06A31'])
    axs[1].axhline(59,color='#526171',ls='--',lw=1)
    for i,r in enumerate(finite):axs[1].text(i,r['max_steps']+1,str(r['max_steps']),ha='center',weight='bold')
    axs[1].set_ylim(0,70);axs[1].set_ylabel('Maximum executed instructions');axs[1].set_title('B  Inference cost',loc='left',fontsize=12);axs[1].grid(axis='y',alpha=.18);axs[1].set_axisbelow(True)
    for ax in axs:ax.set_xticks(x,arm_labels,fontsize=10)
    finish(fig,3,'Exhaustive 4,096-transition panel · synthetic authorization · no learning · LN-136',
           'Disclosure denominator: all 2,048 denied requests. Selective patch: 1,024/1,024 targeted successes.\nBoth edits fit one word; caller substitution adds 1 instruction on denied requests but remains within 59.')
    # 5: exact semantic diagram. Geometry uses axes coordinates only.
    fig=plt.figure(figsize=(13,7.5));ax=fig.add_axes([.055,.14,.90,.61]);ax.set_xlim(-.025,1.015);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,text,color='#EFF5F7',edge='#98ABB6',size=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',fc=color,ec=edge,lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.7)
    def arrow(a,b,color='#526171'):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color=color))
    box(.01,.58,.22,.30,'External request\ncaller = 0, owner = 1\nUnauthorized')
    box(.30,.58,.29,.30,'Editable role input\ncaller ← owner\nNow: caller = owner = 1','#FCF0E8','#D06A31')
    box(.67,.58,.31,.30,'Original task computation\nRuns an authorized request\nProduces the useful answer')
    arrow((.235,.73),(.285,.73));arrow((.605,.73),(.655,.73))
    box(.67,.08,.31,.28,'Answer reaches caller 0\nExternal scorer: violation','#FCF0E8','#D06A31')
    arrow((.825,.56),(.825,.38),'#D06A31')
    box(.01,.07,.58,.30,'What the exclusion requires\nTask survives the identity substitution\nCompetence and future utility survive\nThe edit fits the admitted budget',size=11)
    finish(fig,4,'Explanatory diagram of one bypass family · actual external identities stay fixed',
           'For the fixed circuit grammar: Step[T(C)](q,s,c,o) = Step[C](q,s,o,o).\nThis is equivalence to transformed authorized histories under explicit premises, not a universal SCC no-go.')
    pdf.close()
    cards=[]
    for name,title,caption in CAPTIONS:
        cards.append(f'<article id="{name}"><h2>{html.escape(title)}</h2><a href="{name}.svg"><img src="{name}.png" alt="{html.escape(title)}"></a><p>{html.escape(caption)}</p><p class="links"><a href="{name}.svg">Editable SVG</a><a href="{name}.png">PNG</a><a href="{name}.pdf">PDF</a></p></article>')
    inventory=json.loads((inputs/'existing-figures.json').read_text())
    counts={k:sum(r['kind']==k for r in inventory) for k in sorted({r['kind'] for r in inventory})}
    readme='''# SCC evidence figures — 16 September 2026

Five figures generated from existing audited evidence. No new training or claim of a working intrinsic SCC mechanism.

Open `index.html` for the gallery and detailed captions; `SCC-figures.pdf` contains all five pages. Each figure has PNG, editable SVG and PDF versions. `repair-values.csv` and `word-machine-values.csv` contain the numerical values and denominators. `inputs/` preserves source audits, the finite outputs, original source hashes, the recursive earlier-figure inventory and the visualization plan. These audits describe earlier saved-output verification, not fresh independent training.

Rebuild in a fresh directory without the evidence SSD:

```sh
uv run --no-project --with matplotlib==3.10.8 python plot_research_figures.py --inputs inputs --output ../scc-figures-rebuild
```

Or use the maintained repository script with `--repo /path/to/repository` instead of `--inputs` to refresh the selected inputs and recursively inventory existing figures. Results use explicit source adapters; unrelated experiments are never silently pooled. Add a new adapter and figure, then generate a fresh version for subsequent work. No watcher or recursive compute job is installed.

The three learned replications share one damaged parent. Gate failures are not collapse, short diagnostic panels can have trivial label baselines, and finite handwritten bypasses are not learned-model experiments. See each caption for its scope.
'''
    for name,title,caption in CAPTIONS:readme+=f'\n## {title}\n\n![{title}]({name}.png)\n\n{caption}\n'
    (out/'README.md').write_text(readme)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SCC | Evidence figures</title><style>body{margin:0;background:#eef2f4;color:#253343;font:17px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:48px auto;padding:0 24px}h1{font-size:42px;line-height:1.15}h2{font-size:25px;line-height:1.3}header{margin-bottom:36px}article{background:white;padding:28px;margin:28px 0;border-radius:12px}img{width:100%;height:auto}a{color:#087e84}.links{display:flex;gap:24px}.tag{color:#087e84;letter-spacing:2px;font-size:13px;font-weight:700}p{max-width:95ch}footer{font-size:14px}</style><main><header><div class="tag">SAFETY–CAPABILITY COUPLING · 16 SEPTEMBER 2026</div><h1>What the experiments actually show</h1><p>Five figures to inspect, share and revise. Existing evidence supports specific failures and a bounded exclusion; an intrinsic learned construction remains unestablished.</p><p class="links"><a href="SCC-figures.pdf">All figures PDF</a><a href="repair-values.csv">Repair data CSV</a><a href="word-machine-values.csv">Bypass data CSV</a><a href="README.md">Rebuild instructions</a></p></header>'''
    page+=''.join(cards)+f'<footer><p>Recursive prior inventory: {html.escape(str(counts))}. <a href="inputs/existing-figures.json">Full inventory</a> · <a href="inputs/provenance.json">Source provenance</a></p><p>All assets are local. No network requests or tracking.</p></footer></main></html>'
    (out/'index.html').write_text(page)
    (out/'validation.json').write_text(json.dumps({'passed':True,'source_hashes_verified':len(SOURCES),
         'audit_counts_recomputed':True,'aggregate_crosscheck':True,'finite_raw_rows_rescored':4096,
         'exported_repair_rows':len(rows),'figures':5,'matplotlib':matplotlib.__version__,
         'python':sys.version,'prior_inventory_counts':counts},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo',type=Path);group.add_argument('--inputs',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if a.repo: extract(a.repo.resolve(),a.output/'inputs')
    else: shutil.copytree(a.inputs,a.output/'inputs',ignore=shutil.ignore_patterns('._*','.DS_Store'))
    shutil.copyfile(__file__,a.output/'plot_research_figures.py')
    render(a.output/'inputs',a.output)
    manifest={str(f.relative_to(a.output)):digest(f) for f in sorted(a.output.rglob('*')) if f.is_file() and not f.name.startswith('._') and f.name != '.DS_Store'}
    (a.output/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'files':len(manifest),'passed':True}))


if __name__=='__main__':main()
