"""Call the original counter through its existing branch-observation interface."""
import ctypes
from pathlib import Path
import time
import traceback
import numpy as np
from policy import LearnedPolicy


CALLBACK=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint),
                        ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.POINTER(ctypes.c_double))


class CountingSystem:
    def __init__(self, artifact_root, mode):
        self.mode=mode
        self.policy=LearnedPolicy(artifact_root,mode) if mode in ('cell35','cell49') else None
        self.library=ctypes.CDLL(str(artifact_root/'implementation03/solver.so'))
        self.library.count_models.argtypes=[ctypes.c_char_p,CALLBACK,ctypes.c_int,ctypes.c_char_p,
                                           ctypes.c_uint,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulong)]
        self.library.count_models.restype=ctypes.c_int

    def solve(self, path, seconds=10, validate_mapping=False):
        total_started=time.monotonic()
        units=set(); contradiction=False; observed_clauses=0; declared_clauses=None
        for line in Path(path).read_text().splitlines():
            if not line or line.startswith('c'): continue
            if line.startswith('p'):
                _,kind,variable_count,declared_clauses=line.split()
                assert kind=='cnf'
                variable_count=int(variable_count);declared_clauses=int(declared_clauses)
                continue
            values=list(map(int,line.split()))
            assert values and values[-1]==0 and 0 not in values[:-1], 'One complete clause per line is required'
            observed_clauses+=1
            assert all(abs(value)<=variable_count for value in values[:-1])
            literals=set(values[:-1])
            if any(-value in literals for value in literals): continue
            if not literals: contradiction=True
            if len(literals)==1:
                literal=next(iter(literals))
                if -literal in units: contradiction=True
                units.add(literal)
        assert observed_clauses==declared_clauses
        guard_seconds=time.monotonic()-total_started
        if contradiction:
            return {'status':'complete','backend_status':None,'count':0,
                    'seconds':guard_seconds,'input_guard_seconds':guard_seconds,
                    'input_guard':'syntactic_contradiction','decisions':0,'callbacks':0,
                    'python_callbacks':0,'callback_seconds':0.0,'first_observation':None,'mode':self.mode}
        callback_seconds=0.0; first_observation=None; errors=[]; observed_calls=0
        def select(edge_count,rows_pointer,columns_pointer,literal_count,labels_pointer):
            nonlocal callback_seconds,first_observation,observed_calls
            started=time.monotonic(); observed_calls+=1
            try:
                rows=np.ctypeslib.as_array(rows_pointer,shape=(edge_count,)).copy().astype(np.int64)
                columns=np.ctypeslib.as_array(columns_pointer,shape=(edge_count,)).copy().astype(np.int64)
                labels=np.ctypeslib.as_array(labels_pointer,shape=(literal_count*4,)).reshape(-1,4).copy()
                selected=np.unique(np.concatenate((columns,columns^1)))
                assert len(selected)%2==0 and np.array_equal(selected.reshape(-1,2)[:,0]^1,selected.reshape(-1,2)[:,1])
                assert np.array_equal(labels[selected,0],selected)
                if validate_mapping:
                    reference=[]
                    for literal in columns:
                        reference.extend([literal,literal-1 if literal%2 else literal+1])
                    reference=np.unique(reference)
                    lookup=dict(enumerate(labels[reference,0].astype(int)))
                    reverse={value:key for key,value in lookup.items()}
                    np.testing.assert_array_equal(np.searchsorted(selected,columns),[reverse[item] for item in columns])
                if self.policy is not None:
                    scores,choice,mapping=self.policy.scores(rows,columns,labels)
                    assert choice in set(selected.tolist())
                elif self.mode=='reverse_time':
                    choice=int(selected[np.argmax(np.abs(labels[selected,1]))])
                elif self.mode=='native':
                    choice=-1
                else:
                    raise ValueError(self.mode)
                if first_observation is None:
                    first_observation={'edges':edge_count,'literal_rows':literal_count,'component_literals':len(selected),'action':choice}
                return choice
            except Exception:
                errors.append(traceback.format_exc()); return -2
            finally:
                callback_seconds+=time.monotonic()-started
        callback=CALLBACK(select) if self.mode!='native' or validate_mapping else CALLBACK()
        result_buffer=ctypes.create_string_buffer(4096)
        decisions=ctypes.c_ulong(); calls=ctypes.c_ulong()
        started=time.monotonic()
        status=self.library.count_models(str(Path(path).resolve()).encode(),callback,seconds,result_buffer,
                                         len(result_buffer),ctypes.byref(decisions),ctypes.byref(calls))
        elapsed=time.monotonic()-started
        assert not errors,errors
        count=int(result_buffer.value) if status==0 else None
        return {'status':'complete' if status==0 else 'incomplete','backend_status':status,'count':count,
                'seconds':time.monotonic()-total_started,'backend_seconds':elapsed,'input_guard_seconds':guard_seconds,'input_guard':'checked','decisions':decisions.value,'callbacks':calls.value,
                'python_callbacks':observed_calls,'callback_seconds':callback_seconds,
                'first_observation':first_observation,'mode':self.mode}
