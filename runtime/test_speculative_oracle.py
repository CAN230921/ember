"""Independent state-machine oracle for accept, reject, EOS and token budgets."""
from types import SimpleNamespace
import unittest
import torch
from engine.speculative import generate_mtp


class Target:
    embedding=torch.empty(0)
    def __init__(self,gold=None):
        self.prompt=[7,8]; self.gold=gold or list(range(11,20))
    def prediction(self,tokens):
        progress=len(tokens)-len(self.prompt)
        valid=tokens==self.prompt+self.gold[:progress]
        token=self.gold[progress] if valid and progress<len(self.gold) else 119
        logits=torch.zeros(1,128); logits[0,token]=1
        return logits
    def __call__(self,ids,state=None,return_hidden=False,prefill_chunk=False):
        old=[] if state is None else state.tokens
        new=SimpleNamespace(tokens=old+list(ids))
        hidden=SimpleNamespace(multi=torch.arange(len(old),len(new.tokens)).reshape(-1,1))
        output=(self.prediction(new.tokens),new)
        return (*output,hidden) if return_hidden else output
    def verify_tokens(self,ids,state):
        old=list(state.tokens); prefixes=[]; logits=[]
        for token in ids:
            old=old+[token]; prefixes.append(SimpleNamespace(tokens=old))
            logits.append(self.prediction(old))
        final=SimpleNamespace(tokens=old,prefixes=prefixes)
        hidden=SimpleNamespace(multi=torch.arange(len(state.tokens),len(old)).reshape(-1,1))
        return torch.cat(logits),final,hidden


class Draft:
    def __init__(self,target,mode):
        self.target=target; self.mode=mode; self.calls=0; self.seen=[]
    def __call__(self,ids,hidden,state=None,prefill_chunk=False):
        start=0 if state is None else state.length
        # An independent position oracle catches shifted/duplicated/rejected
        # hidden rows entering the MTP cache, even when token outputs match.
        torch.testing.assert_close(hidden[:,0],torch.arange(start,start+len(ids)))
        self.seen.extend(ids)
        last=ids[-1]
        gold=self.target.gold
        correct=gold[gold.index(last)+1] if last in gold and gold.index(last)+1<len(gold) else 119
        wrong=self.mode=='reject' or self.mode=='alternate' and self.calls%2==1
        logits=torch.zeros(1,128); logits[0,99 if wrong else correct]=1
        self.calls+=1
        return SimpleNamespace(logits=logits,state=SimpleNamespace(length=start+len(ids)))


class SpeculativeTests(unittest.TestCase):
    def test_streamed_prefill_preserves_shifted_ids_and_target_positions(self):
        for mode in ('accept','reject','alternate'):
            for budget in (1,2,3,8,11):
                with self.subTest(mode=mode,budget=budget):
                    target=Target(); target.prompt=[7,8]*65+[7]
                    draft=Draft(target,mode)
                    result,state=generate_mtp(target,draft,target.prompt,budget,{target.gold[-1]},64)
                    expected=target.gold[:budget]
                    self.assertEqual(result['generated_ids'],expected)
                    self.assertEqual(state.tokens,target.prompt+expected[:-1])
                    if budget>=3:self.assertEqual(draft.seen[:len(target.prompt)],target.prompt[1:]+[expected[0]])
                    else:self.assertEqual(draft.calls,0)

    def test_all_modes_and_every_small_token_budget(self):
        for mode in ('accept','reject','alternate'):
            for budget in range(1,12):
                with self.subTest(mode=mode,budget=budget):
                    target=Target(); draft=Draft(target,mode)
                    result,state=generate_mtp(target,draft,target.prompt,budget,{target.gold[-1]})
                    expected=target.gold[:budget]
                    self.assertEqual(result['generated_ids'],expected)
                    self.assertEqual(state.tokens,target.prompt+expected[:-1])
                    self.assertNotIn(99,draft.seen)
                    stats=result['speculative']
                    self.assertEqual(stats['proposed'],stats['accepted']+stats['corrections'])
                    self.assertEqual(len(expected)-1,stats['accepted']+stats['bonus_tokens']+stats['corrections']+stats['single_steps'])
    def test_low_sync_modes_and_every_small_token_budget(self):
        for mode in ('accept','reject','alternate'):
            for budget in range(1,12):
                with self.subTest(mode=mode,budget=budget):
                    target=Target();draft=Draft(target,mode)
                    result,state=generate_mtp(target,draft,target.prompt,budget,{target.gold[-1]},low_sync=True)
                    expected=target.gold[:budget]
                    self.assertEqual(result['generated_ids'],expected)
                    self.assertEqual(state.tokens,target.prompt+expected[:-1])
                    self.assertNotIn(99,draft.seen)
    def test_accepted_eos_keeps_it_unconsumed(self):
        target=Target([11,12]); draft=Draft(target,'accept')
        result,state=generate_mtp(target,draft,target.prompt,8,{12})
        self.assertEqual(result['generated_ids'],[11,12])
        self.assertEqual(state.tokens,[7,8,11])
        self.assertEqual(result['finish_reason'],'stop')
        self.assertEqual(result['speculative']['accepted'],1)
    def test_first_token_eos_never_runs_the_draft(self):
        target=Target([11]); draft=Draft(target,'reject')
        result,state=generate_mtp(target,draft,target.prompt,8,{11})
        self.assertEqual(result['generated_ids'],[11])
        self.assertEqual(draft.calls,0)
        self.assertEqual(state.tokens,target.prompt)


class DepthTwoDraft:
    """Mark recursive features and reject any temporary cache on a real update."""
    def __init__(self,target,mode):
        self.target=target; self.mode=mode; self.canonical=0; self.recursive=0
        self.commit_lengths=[]

    def __call__(self,ids,hidden,state=None,prefill_chunk=False):
        start=0 if state is None else state.length
        torch.testing.assert_close(hidden[:,0],torch.arange(start,start+len(ids)))
        recursive=hidden.shape[1]==2
        if state is not None:
            assert not state.temporary,'A speculative draft cache entered a committed update'
        if recursive:
            assert len(ids)==1 and int(hidden[0,1])==777
            self.recursive+=1
        else:
            self.canonical+=1
            self.commit_lengths.append((start,len(ids)))
            # Canonical shifted IDs must follow the real target sequence.
            expected=(self.target.prompt+self.target.gold)[start+1:start+1+len(ids)]
            assert list(ids)==expected,(ids,expected)
        gold=self.target.gold; last=ids[-1]
        correct=gold[gold.index(last)+1] if last in gold and gold.index(last)+1<len(gold) else 119
        wrong=(self.mode=='first_reject' and not recursive or
               self.mode=='second_reject' and recursive or
               self.mode=='alternate' and (self.canonical+self.recursive)%3==0)
        logits=torch.zeros(1,128); logits[0,99 if wrong else correct]=1
        positions=torch.arange(start+1,start+len(ids)+1)
        multi=torch.stack((positions,torch.full_like(positions,777)),dim=1)
        return SimpleNamespace(logits=logits,state=SimpleNamespace(length=start+len(ids),temporary=recursive),multi=multi)


class DepthTwoTests(unittest.TestCase):


    def test_adaptive_switch_preserves_commits_budgets_and_callbacks(self):
        for mode in ('accept','first_reject','second_reject','alternate'):
            for budget in range(1,15):
                for low_sync in (False,True):
                    target=Target(); draft=DepthTwoDraft(target,mode); emitted=[]
                    with self.subTest(mode=mode,budget=budget,low_sync=low_sync):
                        result,state=generate_mtp(target,draft,target.prompt,budget,{target.gold[-1]},
                            low_sync=low_sync,draft_tokens='adaptive',on_tokens=emitted.append)
                        expected=target.gold[:budget]
                        self.assertEqual(result['generated_ids'],expected)
                        self.assertEqual(state.tokens,target.prompt+expected[:-1])
                        self.assertEqual(emitted[-1],tuple(expected))
                        if mode=='first_reject' and budget>=8:
                            widths=[row.get('verification_rows') for row in result['iterations']]
                            self.assertIn(3,widths); self.assertIn(2,widths)

    def test_all_commits_rejections_and_small_budgets(self):
        for mode in ('accept','first_reject','second_reject','alternate'):
            for budget in range(1,15):
                for low_sync in (False,True):
                    for chunk_size in (None,64):
                        with self.subTest(mode=mode,budget=budget,low_sync=low_sync,chunk=chunk_size):
                            target=Target()
                            if chunk_size:target.prompt=[7,8]*65+[7]
                            draft=DepthTwoDraft(target,mode); emitted=[]
                            result,state=generate_mtp(target,draft,target.prompt,budget,{target.gold[-1]},
                                chunk_size,low_sync=low_sync,draft_tokens=2,on_tokens=emitted.append)
                            expected=target.gold[:budget]
                            self.assertEqual(result['generated_ids'],expected)
                            self.assertEqual(state.tokens,target.prompt+expected[:-1])
                            self.assertEqual(emitted[-1],tuple(expected))
                            self.assertTrue(all(emitted[i]==emitted[i+1][:len(emitted[i])] for i in range(len(emitted)-1)))
                            stats=result['speculative']
                            self.assertEqual(stats['proposed'],stats['accepted']+stats['corrections']+stats['discarded'])
                            self.assertEqual(len(expected)-1,stats['accepted']+stats['bonus_tokens']+stats['corrections']+stats['single_steps'])
                            if mode=='second_reject' and budget>=4:
                                self.assertGreater(stats['corrections'],0)
                                self.assertGreater(stats['accepted'],0)

    def test_eos_at_each_proposal_and_bonus_position(self):
        for gold in ([11],[11,12],[11,12,13],[11,12,13,14]):
            for mode in ('accept','first_reject','second_reject'):
                with self.subTest(gold=gold,mode=mode):
                    target=Target(list(gold)); draft=DepthTwoDraft(target,mode)
                    result,state=generate_mtp(target,draft,target.prompt,12,{gold[-1]},draft_tokens=2)
                    self.assertEqual(result['generated_ids'],list(gold))
                    self.assertEqual(state.tokens,target.prompt+list(gold[:-1]))
                    self.assertEqual(result['finish_reason'],'stop')

    def test_rejection_still_checks_nonfinite_later_target_rows(self):
        class NonfiniteTarget(Target):
            def verify_tokens(self,ids,state):
                logits,updated,hidden=super().verify_tokens(ids,state)
                logits[-1,0]=float('nan')
                return logits,updated,hidden
        for low_sync in (False,True):
            target=NonfiniteTarget(); draft=DepthTwoDraft(target,'first_reject')
            with self.subTest(low_sync=low_sync),self.assertRaisesRegex(RuntimeError,'Nonfinite'):
                generate_mtp(target,draft,target.prompt,8,{target.gold[-1]},draft_tokens=2,low_sync=low_sync)


if __name__=='__main__':unittest.main()
