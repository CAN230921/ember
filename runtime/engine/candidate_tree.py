"""Bounded candidate topology and greedy acceptance, independent of GPU state.

Node zero is the already emitted, not-yet-consumed anchor. A row predicts the
next output token. Every emitted token commits its predicting input node, so
EOS, corrections and the final bonus token remain unconsumed exactly as in
the linear verifier. Proposal scores never decide target acceptance.
"""
from dataclasses import dataclass
import math
import heapq


@dataclass(frozen=True)
class TreeSelection:
    consumed_nodes: tuple
    emitted: tuple
    reason: str


@dataclass(frozen=True)
class CandidateTree:
    tokens: tuple
    parents: tuple

    def __post_init__(self):
        object.__setattr__(self,'tokens',tuple(self.tokens))
        object.__setattr__(self,'parents',tuple(self.parents))
        if not 1<=len(self.tokens)<=8 or len(self.parents)!=len(self.tokens):
            raise ValueError('Expected 1..8 tree nodes including the anchor')
        if any(type(p) is not int for p in self.parents) or self.parents[0]!=-1 or any(type(t) is not int or t<0 for t in self.tokens):
            raise ValueError('Invalid anchor or token IDs')
        edges=set()
        for i in range(1,len(self.tokens)):
            p=self.parents[i]
            if type(p) is not int or not 0<=p<i:raise ValueError('Parents must precede their children')
            key=(p,self.tokens[i])
            if key in edges:raise ValueError('Duplicate token child under one parent')
            edges.add(key)

    def path(self,node):
        if type(node) is not int or not 0<=node<len(self.tokens):raise ValueError('Invalid node')
        path=[]
        while node!=-1:path.append(node);node=self.parents[node]
        return tuple(reversed(path))

    def paths(self):
        nonleaves=set(self.parents[1:])
        return tuple(self.path(i) for i in range(len(self.tokens)) if i not in nonleaves)

    def positions(self,start):
        return tuple(start+len(self.path(i))-1 for i in range(len(self.tokens)))

    def accept_argmax(self,predictions,budget,eos=()):
        values=tuple(predictions)
        if len(values)!=len(self.tokens) or any(type(v) is not int or v<0 for v in values):
            raise ValueError('Expected one target prediction per tree node')
        if type(budget) is not int or budget<1:raise ValueError('Expected a positive output budget')
        children={(p,t):i for i,(p,t) in enumerate(zip(self.parents,self.tokens)) if i}
        node=0;consumed=[];emitted=[]
        while True:
            consumed.append(node);token=values[node];emitted.append(token)
            if token in eos:reason='eos';break
            if len(emitted)==budget:reason='budget';break
            child=children.get((node,token))
            if child is None:
                reason='leaf' if node not in self.parents[1:] else 'miss'
                break
            node=child
        return TreeSelection(tuple(consumed),tuple(emitted),reason)


def tree_from_lattice(anchor,candidates,scores,*,node_budget=8,max_depth=3,width=2):
    """Small beam proposal using the existing DFlash adjacent-candidate scores.

    This is a proposal heuristic, not calibrated target probabilities. Inputs
    may be CPU lists/arrays; GPU-to-CPU transfer is the caller's responsibility.
    At most three draft levels/two active branches fit the reference verifier's
    four-leaf bound. New draft weights are not required.
    """
    if not 1<=node_budget<=8 or not 1<=max_depth<=3 or not 1<=width<=2:
        raise ValueError('Unsupported prototype tree budget/depth/width')
    if len(scores)!=len(candidates):raise ValueError('Candidate/score depths differ')
    tokens=[anchor];parents=[-1];frontier=[(0,0.,0)]
    if node_budget==1:return CandidateTree(tokens,parents)
    for depth in range(min(max_depth,len(candidates))):
        candidates_at_depth=list(candidates[depth]);edges=[]
        for parent,total,previous in frontier:
            row=list(scores[depth][previous])
            if len(row)!=len(candidates_at_depth) or not row or any(not math.isfinite(float(v)) for v in row):
                raise ValueError('Invalid candidate edge scores')
            maximum=max(row);normalizer=maximum+math.log(sum(math.exp(v-maximum) for v in row))
            for j,token in enumerate(candidates_at_depth):
                edges.append((total+row[j]-normalizer,parent,int(token),j))
        edges.sort(key=lambda e:(-e[0],e[1],e[2],e[3]))
        frontier=[];seen=set()
        for score,parent,token,j in edges:
            if (parent,token) in seen:continue
            seen.add((parent,token));tokens.append(token);parents.append(parent)
            frontier.append((len(tokens)-1,score,j))
            if len(frontier)==width or len(tokens)==node_budget:break
        if len(tokens)==node_budget or not frontier:break
    return CandidateTree(tokens,parents)


def tree_from_lattice_best_first(anchor,candidates,scores,*,node_budget=8,max_depth=7,max_leaves=4):
    """Spend the same node budget on the highest-scoring available prefixes.

    Normalized DFlash selector scores are proposal heuristics, not calibrated
    target probabilities. Confident prefixes can extend deeply; uncertain
    prefixes compete for branches. Every parent is allocated before its child.
    Target acceptance and the four-leaf verifier bound remain independent.
    """
    if (type(node_budget) is not int or not 1<=node_budget<=8 or
            type(max_depth) is not int or not 0<=max_depth<=7 or
            type(max_leaves) is not int or not 1<=max_leaves<=4):
        raise ValueError('Unsupported best-first tree budget/depth/leaves')
    if len(scores)!=len(candidates):raise ValueError('Candidate/score depths differ')
    tokens=[anchor];parents=[-1];children=[False];frontier=[];leaves=1;used=set()
    limit=min(max_depth,len(candidates))
    def expand(parent,depth,previous,total):
        if depth>=limit:return
        values=list(candidates[depth])
        if not 0<=previous<len(scores[depth]):raise ValueError('Invalid predecessor score row')
        row=list(scores[depth][previous])
        if len(row)!=len(values) or not row or any(not math.isfinite(float(v)) for v in row):
            raise ValueError('Invalid candidate edge scores')
        maximum=max(row);normalizer=math.log(sum(math.exp(v-maximum) for v in row))
        for index,(token,value) in enumerate(zip(values,row)):
            if isinstance(token,bool) or int(token)!=token or token<0:raise ValueError('Invalid candidate token')
            score=total+(float(value)-maximum)-normalizer
            heapq.heappush(frontier,(-score,parent,int(token),index,depth))
    if node_budget>1:expand(0,0,0,0.)
    while frontier and len(tokens)<node_budget:
        negative,parent,token,index,depth=heapq.heappop(frontier)
        edge=(parent,token)
        if edge in used:continue
        next_leaves=leaves+int(children[parent])
        if next_leaves>max_leaves:continue
        node=len(tokens);tokens.append(token);parents.append(parent);children.append(False)
        children[parent]=True;leaves=next_leaves;used.add(edge)
        expand(node,depth+1,index,-negative)
    return CandidateTree(tokens,parents)
