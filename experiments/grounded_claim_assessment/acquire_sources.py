"""Acquire pinned reader weights and evaluation data without executing either."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request


def main(directory):
    config = json.loads((directory / 'config.json').read_text())
    records = []
    total = 0
    for kind in ('model', 'dataset'):
        prefix = 'datasets/' if kind == 'dataset' else ''
        api_kind = 'datasets' if kind == 'dataset' else 'models'
        identifier = config[kind + '_id']
        revision = config[kind + '_revision']
        metadata_url = f'https://huggingface.co/api/{api_kind}/{identifier}/revision/{revision}?blobs=true'
        with urllib.request.urlopen(metadata_url, timeout=30) as response:
            metadata = json.load(response)
        assert metadata['sha'] == revision
        (directory / (kind + '_metadata.json')).write_text(json.dumps(metadata, indent=2)+'\n')
        inventory = {record['rfilename']: record for record in metadata['siblings']}
        for relative in config[kind + '_files']:
            entry = inventory[relative]
            path = directory / kind / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/{prefix}{identifier}/resolve/{revision}/{relative}'
            with urllib.request.urlopen(url, timeout=45) as response, path.open('xb') as output:
                while block := response.read(1024*1024):
                    total += len(block)
                    if total > config['maximum_bytes']:
                        raise RuntimeError('Declared transfer budget exceeded')
                    output.write(block)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if 'lfs' in entry:
                assert digest == entry['lfs']['sha256']
            elif entry.get('blobId'):
                assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry['blobId']
            records.append({'path':str(path.relative_to(directory)), 'bytes':len(data), 'sha256':digest})
    receipt = {'status':'complete','total_bytes':total,'files':records,'model_executed':False,'neural_training':False}
    (directory/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='files'}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
