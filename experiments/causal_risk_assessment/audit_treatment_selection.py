"""Count treatment-selection pairs without loading embedded causal model code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    for name in ('extractor','datasets','output'):
        parser.add_argument('--'+name,required=True)
    arguments=parser.parse_args()
    sys.path.insert(0,arguments.extractor)
    from evaluate_released_cases import extract_dataset
    records=[]
    for path in sorted(Path(arguments.datasets).glob('data/prior_sampling/*/*.pkl')):
        arrays,receipt=extract_dataset(path)
        pairs,counts=np.unique(np.column_stack([arrays['x_obs'][:,0],arrays['x_int'][:,0]]),axis=0,return_counts=True)
        records.append({'case':path.stem,'rows':receipt['rows'],'pairs':pairs.tolist(),'counts':counts.tolist(),
            'changed':int(np.sum(arrays['x_obs'][:,0]!=arrays['x_int'][:,0])),'source_sha256':receipt['source_sha256']})
    result={'case_count':len(records),'rows':sum(record['rows'] for record in records),'changed':sum(record['changed'] for record in records),'records':records}
    Path(arguments.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='records'}))

if __name__=='__main__':
    main()
