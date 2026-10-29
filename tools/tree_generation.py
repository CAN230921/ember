"""Bounded experimental greedy generation using the current DFlash2 lattice.

For validation only. Single request, no prefix-cache publishing, no HTTP or
replacement of the accepted runner. Every output is chosen by target argmax.
"""
import torch

from engine.candidate_tree import CandidateTree
from engine.dflash_draft import LAYERS


@torch.inference_mode()
def generate_tree(runtime,ids,budget,*,policy=None):
    if type(budget) is not int or budget<1:raise ValueError('Expected positive output budget')
    engine=runtime.engine;draft=runtime.batch_runner.dflash
    if draft is None:raise ValueError('Tree validation requires the current DFlash2 draft')
    taps={}
    def observe(index,value):
        if isinstance(index,int) and index in LAYERS:taps[index]=draft.tap(*value)
    logits,state=engine(ids,observer=observe,prefill_chunk=True)
    output=[int(logits[-1].argmax())];rounds=[]
    ctx=draft.new_context()
    draft.extend(ctx,torch.cat([taps[i] for i in LAYERS],-1))
    while len(output)<budget and output[-1] not in runtime.eos:
        remaining=budget-len(output)
        tree=(CandidateTree((output[-1],),(-1,)) if remaining==1
              else draft.propose_tree(ctx,output[-1],policy=policy))
        taps={}
        tx=engine.verify_tree(tree,state,observe)
        try:
            selected=tx.select(remaining,runtime.eos)
            state=tx.commit(selected)
        finally:
            if tx.status=='open':tx.discard()
        indices=torch.tensor(selected.consumed_nodes,device=engine.embedding.device,dtype=torch.long)
        features=torch.cat([taps[i].index_select(0,indices) for i in LAYERS],-1)
        draft.push(ctx,features)
        if ctx.length!=len(state.tokens):raise RuntimeError('DFlash tree feature history is misaligned')
        output.extend(selected.emitted)
        rounds.append(dict(nodes=len(tree.tokens),consumed_nodes=selected.consumed_nodes,
                           emitted=len(selected.emitted),reason=selected.reason,
                           depth=max(len(p) for p in tree.paths())-1,leaves=len(tree.paths())))
    return dict(generated_ids=output,rounds=rounds,
                finish_reason='stop' if output[-1] in runtime.eos else 'length')
