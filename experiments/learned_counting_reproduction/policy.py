"""Use pinned policy class definitions with an explicit sum-message graph adapter."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


SETTINGS = dict(state_dim=0, embedding_dim=32, vemb_dim=32, cemb_dim=32,
                vlabel_dim=0, clabel_dim=0, policy_dim1=256, policy_dim2=64,
                max_iters=2, max_variables=1000000, state_bn=False, use_bn=False,
                entropy_alpha=0, lambda_value=0, lambda_disallowed=0, lambda_aux=0,
                policy_non_linearity='F.relu', non_linearity='F.relu', print_every=0,
                sharp_encoder_type='GINEncoder', sharp_decode_class='SudokuModel1',
                sharp_decode=False, sharp_decode_size=9, sharp_decoded_emb_dim=32,
                sharp_add_embedding=True, sharp_add_labels=False, sharp_emb_dim=32,
                cp_num_layers=3)


class PolicyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.settings=SETTINGS.copy()


class MessageFunctions:
    @staticmethod
    def copy_src(source, message):
        return source, message

    @staticmethod
    def sum(message, destination):
        return message, destination


class EdgeView:
    def __init__(self, graph, source, destination, source_indices, destination_indices):
        self.graph=graph; self.source=source; self.destination=destination
        self.source_indices=source_indices; self.destination_indices=destination_indices

    def update_all(self, message, reduction):
        features=self.graph.nodes[self.source].data[message[0]]
        size=self.graph.number_of_nodes(self.destination)
        if self.graph.dense:
            incidence=torch.zeros((size,len(features)),dtype=features.dtype)
            incidence[self.destination_indices,self.source_indices]=1
            aggregate=incidence @ features
        else:
            aggregate=torch.zeros((size,features.shape[1]),dtype=features.dtype)
            aggregate.index_add_(0,self.destination_indices,features[self.source_indices])
        self.graph.nodes[self.destination].data[reduction[1]]=aggregate


class MessageGraph:
    def __init__(self, features, adjacency, dense):
        self.dense=dense
        coordinates=adjacency.coalesce().indices()
        rows,columns=coordinates[0],coordinates[1]
        self.sizes={'literal':adjacency.shape[1],'clause':adjacency.shape[0]}
        self.nodes={'literal':SimpleNamespace(data={'literal_feats':features}),
                    'clause':SimpleNamespace(data={})}
        self.edges={'l2c':EdgeView(self,'literal','clause',columns,rows),
                    'c2l':EdgeView(self,'clause','literal',rows,columns)}

    def number_of_nodes(self, kind):
        return self.sizes[kind]

    def __getitem__(self, relation):
        return self.edges[relation]


def load_definitions(path, names, namespace):
    syntax=ast.parse(path.read_text())
    selected=[node for node in syntax.body if isinstance(node,ast.ClassDef) and node.name in names]
    assert {node.name for node in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


class LearnedPolicy:
    def __init__(self, artifact_root, name, dense=False):
        source=artifact_root/'acquisition01/upstream'
        additional=artifact_root/'adapter01/upstream'
        namespace=dict(torch=torch,nn=nn,F=functional,np=np,nn_init=nn.init,
                       PolicyBase=PolicyBase,CnfSettings=lambda:SETTINGS.copy(),
                       fn=MessageFunctions,TimerStat=nullcontext,
                       undensify_obs=lambda observation:observation,
                       graph_from_adj=lambda features,clauses,adjacency:MessageGraph(features,adjacency,dense))
        load_definitions(source/'common_components.py',{'MLPModel'},namespace)
        load_definitions(additional/'sudoku_models.py',{'SudokuModel1'},namespace)
        load_definitions(source/'dgl_encoders.py',{'DGLEncoder','GINEncoder'},namespace)
        load_definitions(source/'rllib_sharp_models.py',{'SharpModel'},namespace)
        self.model=namespace['SharpModel']()
        weights=np.load(artifact_root/'adapter01'/(name+'_weights.npy'),allow_pickle=False)
        self.layout=[]; offset=0; state=self.model.state_dict()
        for key,value in state.items():
            size=value.numel()
            self.layout.append({'name':key,'shape':list(value.shape),'offset':offset,'count':size})
            state[key]=torch.from_numpy(weights[offset:offset+size].copy()).view_as(value)
            offset+=size
        assert offset==len(weights)==82098
        self.model.load_state_dict(state,strict=True)
        self.model.eval().requires_grad_(False)

    def scores(self, rows, columns, labels):
        selected=np.unique(np.concatenate((columns,np.bitwise_xor(columns,1))))
        assert np.array_equal(labels[selected,0],selected)
        local_columns=np.searchsorted(selected,columns)
        assert np.array_equal(selected[local_columns],columns)
        coordinates=torch.from_numpy(np.stack((rows,local_columns)).astype(np.int64))
        adjacency=torch.sparse_coo_tensor(coordinates,torch.ones(len(rows)),
                                        (int(max(rows))+1,len(selected))).coalesce()
        observation=SimpleNamespace(ground=torch.from_numpy(labels[selected,1:].copy()).float(),
                                    cmat=adjacency,ext_data=(None,[]))
        with torch.inference_mode():
            scores,_=self.model(observation,None,None)
            assert torch.isfinite(scores).all()
            # Match the released TorchCategoricalArgmax, including normalization.
            action=int(torch.distributions.Categorical(logits=scores).logits.argmax())
        return scores.numpy().reshape(-1),int(selected[action]),selected
