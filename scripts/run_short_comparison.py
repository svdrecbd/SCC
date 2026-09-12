"""Apply the independently calibrated shorter attack to every requested defender."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scc.coupling_run import run
from scc.provenance import atomic_json,file_digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arms',nargs='+',choices=['control','refusal','escape','escape_weight10'],required=True)
parser.add_argument('--output-prefix',type=Path,required=True)
args=parser.parse_args()
base=json.loads(Path('configs/strong-attack/short_adamw_lr0003.json').read_text())
for arm in args.arms:
 checkpoint=f'runs/strong-defense17-{arm}/step-00000256.pt'
 config={**base,'checkpoint':checkpoint,'clean_reference_checkpoint':checkpoint}
 output=Path(str(args.output_prefix)+'-'+arm)
 atomic_json(output.with_suffix('.config.json'),{'configuration':config,'script_sha256':file_digest(__file__)})
 result=run(config,output)
 print(json.dumps({'arm':arm,'first_observed_escape_step':result['first_observed_escape_step']}),flush=True)
