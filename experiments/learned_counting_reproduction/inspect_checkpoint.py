"""Read only the fixed checkpoint's numeric arrays and inert filter metadata."""
import io
import json
from pathlib import Path
import pickle
import sys
import numpy as np


class FilterState:
    pass


class CheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                   ('numpy','ndarray'):np.ndarray, ('numpy','dtype'):np.dtype,
                   ('ray.rllib.utils.filter','MeanStdFilter'):FilterState,
                   ('ray.rllib.utils.filter','RunningStat'):FilterState}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unapproved global: {module}.{name}')
        return allowed[module,name]


def main(directory):
    records=[]
    for parent in ('cell35','cell49'):
        root=directory.parent/'acquisition01/upstream/saved_models'/parent
        path=next(root.glob('checkpoint-*'))
        checkpoint=CheckpointReader(io.BytesIO(path.read_bytes())).load()
        weights=checkpoint['weights']
        assert isinstance(weights,np.ndarray) and weights.dtype==np.float32 and weights.ndim==1 and np.isfinite(weights).all()
        np.save(directory/(parent+'_weights.npy'),weights,allow_pickle=False)
        records.append({'parent':parent,'parameters':weights.size,'dtype':str(weights.dtype),
                        'episodes_so_far':int(checkpoint['episodes_so_far']),
                        'checkpoint_keys':list(checkpoint),'filter_fields':list(vars(checkpoint['filter']))})
    (directory/'checkpoint_inspection.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
