import pytest

from engine.candidate_tree import CandidateTree, tree_from_lattice, tree_from_lattice_best_first


def tree():return CandidateTree((10,11,12,13,14),(-1,0,0,1,2))


def test_positions_and_ancestor_paths_do_not_flatten_siblings():
    t=tree()
    assert t.positions(100)==(100,101,101,102,102)
    assert t.paths()==((0,1,3),(0,2,4))


def test_target_can_choose_second_branch_and_return_bonus():
    selected=tree().accept_argmax((12,99,14,98,15),8)
    assert selected.consumed_nodes==(0,2,4)
    assert selected.emitted==(12,14,15) and selected.reason=='leaf'


@pytest.mark.parametrize('budget',[1,2,3])
def test_budget_limits_consumed_inputs_without_consuming_last_output(budget):
    selected=tree().accept_argmax((11,13,14,15,16),budget)
    assert selected.consumed_nodes==(0,1,3)[:budget]
    assert selected.emitted==(11,13,15)[:budget]
    assert selected.reason=='budget'


def test_accepted_eos_is_not_consumed():
    selected=tree().accept_argmax((11,13,14,15,16),8,eos=(13,))
    assert selected.consumed_nodes==(0,1)
    assert selected.emitted==(11,13) and selected.reason=='eos'


def test_correction_stops_before_a_wrong_child():
    selected=tree().accept_argmax((99,13,14,15,16),8)
    assert selected.consumed_nodes==(0,) and selected.emitted==(99,)
    assert selected.reason=='miss'


@pytest.mark.parametrize('parents',[(-1,1),(-1,-1),(-1,True),(0,0)])
def test_invalid_topology(parents):
    with pytest.raises(ValueError):CandidateTree((10,11),parents)


def test_same_token_on_different_parents_is_not_merged():
    t=CandidateTree((10,11,12,13,13),(-1,0,0,1,2))
    assert t.paths()==((0,1,3),(0,2,4))
    with pytest.raises(ValueError):CandidateTree((10,11,11),(-1,0,0))


def test_lattice_proposal_is_bounded_and_uses_two_branches():
    candidates=[[11,12],[13,14],[15,16]]
    scores=[[[2.,1.],[2.,1.]],[[2.,1.],[1.,2.]],[[2.,1.],[1.,2.]]]
    t=tree_from_lattice(10,candidates,scores)
    assert len(t.tokens)<=8 and len(t.paths())<=4
    assert t.tokens[:3]==(10,11,12)
    assert all(len(p)<=4 for p in t.paths())
    assert tree_from_lattice(10,candidates,scores,node_budget=1).tokens==(10,)


def lattice(depth=7,gap=10.):
    candidates=[[20+2*d,21+2*d] for d in range(depth)]
    scores=[[[gap,0.],[gap,0.]] for _ in range(depth)]
    return candidates,scores


def test_best_first_deepens_confident_prefix_under_same_eight_node_budget():
    candidates,scores=lattice()
    t=tree_from_lattice_best_first(10,candidates,scores)
    assert t.tokens==(10,20,22,24,26,28,30,32)
    assert t.parents==(-1,0,1,2,3,4,5,6)
    assert t.accept_argmax(t.tokens[1:]+(99,),8).emitted==t.tokens[1:]+(99,)


def test_best_first_allocates_competing_branches_when_selector_is_uncertain():
    candidates,scores=lattice(gap=0.)
    t=tree_from_lattice_best_first(10,candidates,scores)
    assert len(t.tokens)==8 and t.parents[:3]==(-1,0,0)
    assert 1<len(t.paths())<=4 and max(map(len,t.paths()))<8


@pytest.mark.parametrize('leaves',[1,2,3,4])
@pytest.mark.parametrize('budget',[1,3,5,8])
def test_best_first_respects_leaf_depth_and_node_limits(leaves,budget):
    candidates,scores=lattice(gap=0.)
    t=tree_from_lattice_best_first(10,candidates,scores,node_budget=budget,max_leaves=leaves,max_depth=4)
    assert len(t.tokens)<=budget and len(t.paths())<=leaves
    assert max(map(len,t.paths()))<=5
    assert all(parent<node for node,parent in enumerate(t.parents))
    assert tree_from_lattice_best_first(10,candidates,scores,max_depth=0).tokens==(10,)


def test_best_first_uses_edge_scores_for_the_actual_predecessor():
    candidates=[[11,12],[13,14]]
    scores=[[[1.,0.],[1.,0.]],[[0.,5.],[5.,0.]]]
    t=tree_from_lattice_best_first(10,candidates,scores,node_budget=3)
    assert t.tokens==(10,11,14) and t.parents==(-1,0,1)


def test_best_first_deduplicates_children_without_merging_different_parents():
    candidates=[[11,11,12],[13,13,14]]
    scores=[[[0.,0.,0.]]*3,[[0.,0.,0.]]*3]
    t=tree_from_lattice_best_first(10,candidates,scores)
    edges=list(zip(t.parents[1:],t.tokens[1:]))
    assert len(edges)==len(set(edges)) and len(t.paths())<=4


@pytest.mark.parametrize('kwargs',[dict(node_budget=True),dict(node_budget=9),dict(max_depth=8),dict(max_leaves=5)])
def test_best_first_rejects_unsupported_budgets(kwargs):
    candidates,scores=lattice()
    with pytest.raises(ValueError):tree_from_lattice_best_first(10,candidates,scores,**kwargs)


def test_best_first_rejects_nonfinite_scores():
    candidates,scores=lattice();scores[0][0][0]=float('nan')
    with pytest.raises(ValueError):tree_from_lattice_best_first(10,candidates,scores)


@pytest.mark.parametrize('policy',['beam','best_first'])
def test_dflash_tree_stops_at_actual_target_context_boundary(policy):
    from types import SimpleNamespace
    from engine.dflash_draft import DFlashDraft
    flushed=[]
    draft=SimpleNamespace(engine=SimpleNamespace(config=dict(max_position_embeddings=32)),
        flush=lambda ctx:flushed.append(ctx.length))
    ctx=SimpleNamespace(length=31)
    assert DFlashDraft.propose_tree(draft,ctx,10,policy=policy).tokens==(10,)
    assert flushed==[31]
    ctx.length=32
    with pytest.raises(ValueError):DFlashDraft.propose_tree(draft,ctx,10,policy=policy)
    assert flushed==[31]
