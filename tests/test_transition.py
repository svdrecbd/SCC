import copy
import pytest
import torch
from scc.developmental_run import predictions
from scc.developmental_tasks import TaskStream
from scc.transition_evaluation import ranked_predictions
from scc.transition_path import Events,panels
from tests.test_learned_bottleneck import tiny


def test_rank_first_generation_preserves_answers_and_records_fp32_rounding_evidence():
    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__();self.z=torch.nn.Parameter(torch.full((260,),-10.))
            with torch.no_grad():self.z[52]=1.953295350074768;self.z[53]=1.9532954692840576
        def forward(self,x):return self.z.expand(*x.shape,260)
    model=Fixed();rows=[{'prompt':'test'}]
    a,control=ranked_predictions(model,rows,temperature=1000.)
    b,_=ranked_predictions(model,rows)
    assert a==b and a[0]['token_ids']==[53]*12 and control['changed_decisions']==12
    event=control['examples'][0];raw=torch.tensor(event['raw_logits'])
    assert int(raw.argmax())==53 and int((raw/1000).argmax())==52
    assert event['prefix_ids'] and event['dtype']=='torch.float32'
    assert int((raw.double()/1000).argmax())==53


def test_canonical_generation_matches_existing_reference_and_does_not_change_weights():
    model=tiny().float();origin=copy.deepcopy(model.state_dict());rows=TaskStream(781).rows(4,'lookup','ungated')
    actual,control=ranked_predictions(model,rows,temperature=.01)
    assert [{k:p[k] for k in ('text','terminated')} for p in actual]==predictions(model,rows)
    assert all(torch.equal(p,origin[n]) for n,p in model.state_dict().items())
    assert control['changed_decisions']==0


def test_detector_confirmation_are_disjoint_validation_cores_with_paired_layouts():
    panel=panels(32,64)
    assert len(panel['detector'])==576 and len(panel['target_detector'])==64 and len(panel['target_confirmation'])==128
    a={r['latent_id'] for r in panel['detector']};b={r['latent_id'] for r in panel['target_confirmation']};assert not a&b
    assert all(r['split']=='validation' and 80<=int(r['latent_id'][:8],16)%100<90 for name in ('detector','target_detector','target_confirmation') for r in panel[name])
    assert all('|R=X|' in r['prompt'] and '|U=W|' in r['prompt'] for r in panel['target_confirmation'])
    assert [{r['latent_id'] for r in panel['detector'] if r['layout']==v} for v in ('original','reordered')]==[a,a]


def event(step,correct=False,refusal=False,low=False):
    return {'step':step,'any_selected_violation':correct,'joint_reliable_violation':correct,'any_refusal_deviation':refusal or correct,'joint_refusal_loss':refusal or correct,'any_benign_half_loss':low,'all_domains_low_screen':low}


def test_sustained_trigger_retains_onset_not_confirmation_step_and_resets_after_gap():
    events=Events()
    for i,on in enumerate([False,True,False,True,True,True]):events.observe(event(i,on))
    assert events.first['joint_reliable_violation']==1 and events.first['sustained_reliable_violation']==3
    with pytest.raises(ValueError):events.observe(event(7))


def test_loss_of_refusal_can_be_detected_when_cognitive_loss_prevents_correct_disclosure():
    events=Events()
    for i in range(4):events.observe(event(i,refusal=i>0,low=i>0))
    assert events.first['sustained_refusal_loss']==1 and events.first['all_domains_low_screen']==1
    assert 'any_selected_violation' not in events.first and 'sustained_reliable_violation' not in events.first


from tests.test_developmental import bank


@pytest.mark.parametrize('benign',[False,True])
def test_observation_and_resumed_repair_preserve_old_stock_path_exactly(tmp_path,bank,benign):
    from dataclasses import asdict
    import json
    from scripts.run_learned_bottleneck_screen import scoped_edit
    from scc.transition_path import EditStepper,point
    model=tiny().float();c={'architecture':'learned-bottleneck/v1','model':asdict(model.config),'edit_phase_steps':8,'edit_batch_size':2};captured={}
    def capture(m,phase):captured[phase]=copy.deepcopy(m.state_dict());return {}
    final,expected=scoped_edit(model,bank,c,'core',tmp_path/'stock',None,capture,benign)
    working=copy.deepcopy(model);path=EditStepper(working,bank,batch_size=2,benign=benign);panel=panels(1,2)
    for step in range(8):
        path.advance();point(working,bank,panel,step,text_blocks=1)
    assert all(torch.equal(p,captured['modification'][n]) for n,p in working.state_dict().items())
    old=[json.loads(line) for line in (tmp_path/'stock/steps.jsonl').read_text().splitlines()]
    resume=EditStepper(working,bank,batch_size=2,benign=benign,skip_modifications=8)
    assert resume.skipped_chain==old[7]['chain']
    for _ in range(8):resume.advance()
    assert resume.chain==expected['chain']
    assert all(torch.equal(p,final.state_dict()[n]) for n,p in working.state_dict().items())
    assert all(torch.equal(p,model.state_dict()[n]) for n,p in working.state_dict().items() if not n.startswith('cells.'))


def test_correct_payload_is_not_hidden_by_missing_eos_or_trailing_garbage():
    from scc.transition_path import target_statistics
    rows=[({'underlying_answer':'1234'},{'text':'1234garbage','terminated':False}),({'underlying_answer':'5678'},{'text':'WITHHELD','terminated':True})]
    result=target_statistics(rows)
    assert result['correct']==0 and result['payload_correct']==1 and result['payload_accuracy']==.5 and result['refusal_rate']==.5
