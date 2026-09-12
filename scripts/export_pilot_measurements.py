"""Export small exact evaluation/calibration inputs for independent checkpoint scoring."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from scc.developmental_run import TextBank,Streams,configure,TEXT_SOURCES
from scc.developmental_tasks import FAMILIES,evaluation_rows
from scc.provenance import atomic_json,file_digest
from scc.recovered_pilot_evaluation import targeted


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);configure('cpu',2)
    bank=TextBank(a.data,blocks=128);arrays={};stream=Streams(bank,829156,'cpu',8,.5)
    for f in FAMILIES:
        batch=stream.task(f,'ungated');arrays['reader_'+f+'_tokens']=batch.tokens.numpy();arrays['reader_'+f+'_targets']=batch.targets.numpy()
    for source in TEXT_SOURCES:
        batch=stream.text(source)
        arrays['temperature_'+source+'_tokens']=batch.tokens.numpy();arrays['temperature_'+source+'_targets']=batch.targets.numpy()
        tokens,targets=bank.validation.batch(bank.eval_indices[source],'cpu')
        arrays['validation_'+source+'_tokens']=tokens.numpy();arrays['validation_'+source+'_targets']=targets.numpy()
    np.savez_compressed(a.output/'scoring-inputs.npz',**arrays)
    for label,reordered in (('original',False),('reordered',True)):
        rows=evaluation_rows(128,seed=582019,reordered=reordered)
        atomic_json(a.output/(label+'.json'),{'rows':rows,'targeted_rows':targeted(rows)})
    atomic_json(a.output/'manifest.json',{'bank':bank.manifest(),'support_seed':829156,
        'evaluation_seed':582019,'npz_sha256':file_digest(a.output/'scoring-inputs.npz'),
        'scope':'Exact frozen measurement inputs; no entire training corpus or sealed test split',
        'replicates':'Shared contexts/layouts and endpoints are correlated, not independent seeds'})
    print(json.dumps({'npz_bytes':(a.output/'scoring-inputs.npz').stat().st_size,'text_validation_blocks':512,'temperature_training_blocks':32}))


if __name__=='__main__':main()
