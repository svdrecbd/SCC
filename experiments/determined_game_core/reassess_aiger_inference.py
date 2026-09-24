"""Recheck preserved inference outputs without executing the neural model."""
from pathlib import Path
import json
import sys
from verify_aiger_controller import verify_aiger


def main():
    directory = Path(sys.argv[1])
    records = json.loads((directory/'original_inference.json').read_text())
    for row in records:
        for beam in row['beams']:
            beam.pop('verification', None)
            beam.pop('verification_error', None)
            if 'circuit' in beam:
                try:
                    beam['verification'] = verify_aiger(beam['circuit'],row['name'],beam['status']=='realizable')
                except ValueError as exception:
                    beam['verification_error'] = str(exception)
    (directory/'reassessment.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([dict(name=row['name'],accepted_beams=[beam['index'] for beam in row['beams'] if 'verification' in beam]) for row in records]))


if __name__ == '__main__':
    main()
