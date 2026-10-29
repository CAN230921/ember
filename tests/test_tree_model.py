from types import SimpleNamespace

import numpy as np
import torch

from engine.candidate_tree import CandidateTree
from engine.tree_model import attention_nodes,ple_nodes


def tree():return CandidateTree((10,11,12,13,14),(-1,0,0,1,2))


def test_unique_gdn_projection_is_shared_but_recurrence_follows_parents():
    calls=[]
    def project(x,**kwargs):calls.append(len(x));return x,x
    def core(x,ba,previous,prefill):return x+previous,float(x[0,0])+previous
    attention=SimpleNamespace(project=project,run_core=core,out_linear=lambda x:x)
    layer=SimpleNamespace(layer_type='linear_attention',attention=attention)
    output,states=attention_nodes(layer,torch.arange(5).reshape(5,1).float(),None,tree(),10.,100)
    assert calls==[5] and states==[10.,11.,12.,14.,16.]
    assert output.flatten().tolist()==states


def test_tree_ple_hash_and_convolution_histories_exclude_siblings():
    contexts=[]
    class Hasher:
        ngram_size=3;eos=0
        def __call__(self,ids,starts,context):
            contexts.extend(context.tolist());assert starts.tolist()==[0,1,2,3,4,5]
            return ids[:,None]
    def core(hidden,embeddings,previous):
        state=previous+tuple(embeddings.flatten().tolist())
        return embeddings,state
    ple=SimpleNamespace(hasher=Hasher(),core=core,
        table=SimpleNamespace(lookup=lambda ids,device:torch.tensor(ids,dtype=torch.float32)))
    _,states=ple_nodes(SimpleNamespace(ple=ple),torch.zeros(5,1),tree(),[1,2,3],())
    assert contexts==[[2,3],[3,10],[3,10],[10,11],[10,12]]
    assert states[3]==(10.,11.,13.) and states[4]==(10.,12.,14.)


def test_ple_reads_only_the_ngram_tail_even_for_long_histories_and_eos():
    from engine.ops.ple_lookup import NGramHasher
    class TailHistory(list):
        def __iter__(self):raise AssertionError('PLE scanned irrelevant full history')
        def __getitem__(self,index):
            assert isinstance(index,slice) and index.start==-2
            return super().__getitem__(index)
    hasher=NGramHasher(np.array([17,19,23]),np.array([127,131]),np.array([0,127]),0)
    ple=SimpleNamespace(hasher=hasher,table=SimpleNamespace(lookup=lambda ids,device:torch.from_numpy(ids)),
        core=lambda hidden,embeddings,previous:(embeddings,previous))
    layer=SimpleNamespace(ple=ple);hidden=torch.zeros(5,1)
    full=TailHistory([7]*262134+[0,9])
    actual,_=ple_nodes(layer,hidden,tree(),full,None)
    expected,_=ple_nodes(layer,hidden,tree(),[0,9],None)
    assert torch.equal(actual,expected)
