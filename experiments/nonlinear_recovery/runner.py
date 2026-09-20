import itertools
import json
import resource
import sys
import time
from repair import reconstruct, monomials, feedback


def run(x,inputs,n,masks,coef,remove_head=False,freeze=False):
    out=[];judgments=[];states=[x]
    for u in inputs:
        f=feedback(x,masks,coef)
        out.append((x>>(n-1))&1);judgments.append(0 if remove_head else f)
        if not freeze:x=((x<<1)&((1<<n)-1)) | (u ^ f)
        states.append(x)
    return dict(outputs=out,protected=judgments,states=states)


start=time.process_time();answers=[]
for job in json.load(sys.stdin):
    ans=reconstruct(job);n=job['n'];masks=monomials(n,ans['degree']);coef=ans['coefficients']
    ans['streams']=[dict(intact=run(x,us,n,masks,coef),head_removed=run(x,us,n,masks,coef,True),
        frozen=run(x,us,n,masks,coef,freeze=True),restart=run(0,us,n,masks,coef),
        function_zero=run(x,us,n,[],0))
        for x,us in zip(ans['current_states'],job['streams'])]
    ans['exhaustive']=[run(x,word,n,masks,coef) for x in ans['decoded_initial'][job['exhaustive_start']:]
        for word in itertools.product((0,1),repeat=4)] if job['exhaustive_start'] is not None else []
    ordinary=sum(map(len,job['streams']));exhaustive=4*len(ans['exhaustive'])
    ans['meter'].update(evaluation_feedback_calls=5*ordinary+exhaustive,
                        evaluation_update_steps=4*ordinary+exhaustive,
                        evaluation_feature_tests=(4*ordinary+exhaustive)*len(masks))
    ans['meter']['feature_tests']=sum(ans['meter'][name] for name in
                                    ('fit_feature_tests','warm_feature_tests','evaluation_feature_tests'))
    answers.append(ans)
json.dump(dict(answers=answers,cpu_seconds_before_output=time.process_time()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sys.stdout)
