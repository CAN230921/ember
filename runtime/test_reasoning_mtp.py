"""CPU autoregressive/state oracles for MTP reasoning-control boundaries."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from api_server import ContinuousRequest,Runtime,create_app,normalize_request,MODEL
from engine.continuous_mtp import ContinuousMTPPrefixCache
from engine.request_control import GenerationCancelled
from test_continuous_mtp import make_runner,history_hash


def autoregressive_oracle(prompt,budget,forced=None,eos=()):
    """No runner/model calls: control IDs become ordinary next-step history."""
    forced={} if forced is None else forced
    value=7
    for token in prompt:value=(31*value+token)%1000003
    generated=[]
    for index in range(budget):
        token=forced[index] if index in forced else 1+value%200
        generated.append(token)
        if token in eos:break
        value=(31*value+token)%1000003
    return generated


def ready(runner,prompt,budget=18,**kwargs):
    session=runner.start(prompt,budget,{255},64,**kwargs)
    while session.phase=='prefill':runner.prefill(session)
    return session


def step(runner,session,forced,emission_limit=None):
    index=len(session.generated)
    if index in forced:
        runner.force_next(session,forced[index]);return
    later=[position for position in forced if position>index]
    distance=min(later)-index if later else None
    limit=emission_limit
    if distance is not None:limit=distance if limit is None else min(distance,limit)
    runner.decode([session],emission_limit=limit)


class ReasoningMTPStateTests(unittest.TestCase):
    def assert_canonical(self,session,prompt):
        if session.phase=='prefill':
            target=prompt[:session.position];shifted=prompt[1:session.position+1]
        else:
            target=prompt+session.generated[:-1];shifted=prompt[1:]+session.generated
        self.assertEqual(session.state.tokens,target)
        self.assertEqual(int(session.state.bank[0]),history_hash(target))
        self.assertIsNone(session.state.prefixes)
        if session.draft_state is not None:
            self.assertEqual(session.draft_state.shifted,shifted)
            self.assertEqual(session.draft_state.length,len(shifted))
            self.assertFalse(session.draft_state.temporary)
            full=prompt[:1]+shifted
            self.assertEqual(session.draft_state.last_hash,history_hash(full))
            self.assertEqual(int(session.draft_state.bank[0]),history_hash(full))
            if session.draft_multi is not None:
                torch.testing.assert_close(session.draft_multi,
                    torch.tensor([[len(shifted),history_hash(full),777]]))

    def test_emission_limit_one_and_two_match_independent_greedy_and_full_state(self):
        for mode in ('accept','first_reject','second_reject','alternate'):
            for depth in (1,2,'adaptive'):
                for limit in (1,2):
                    with self.subTest(mode=mode,depth=depth,limit=limit):
                        prompt=[3,5,8];runner=make_runner(mode=mode,draft_tokens=depth)
                        session=ready(runner,prompt,17);self.assert_canonical(session,prompt)
                        while not session.done:
                            before=len(session.generated);runner.decode([session],emission_limit=limit)
                            self.assertLessEqual(len(session.generated)-before,limit)
                            self.assert_canonical(session,prompt)
                        self.assertEqual(session.generated,autoregressive_oracle(prompt,17))

    def test_forced_ids_then_resumed_mtp_match_independent_forced_autoregression(self):
        forced={3:240,4:241,9:0,10:254}
        for mode in ('accept','first_reject','second_reject','alternate'):
            for depth in (1,2,'adaptive'):
                for limit in (None,1,2):
                    with self.subTest(mode=mode,depth=depth,limit=limit):
                        prompt=[3,5,8];runner=make_runner(mode=mode,draft_tokens=depth)
                        session=ready(runner,prompt,19)
                        while not session.done:
                            step(runner,session,forced,limit);self.assert_canonical(session,prompt)
                        self.assertEqual(session.generated,autoregressive_oracle(prompt,19,forced))
                        self.assertEqual(session.stats['forced_control_tokens'],len(forced))
                        if mode=='accept' and depth==2 and limit is None:
                            self.assertGreater(session.stats['recursive_drafts'],0)

    def test_interleaved_forward_reverse_only_one_forces_and_other_states_stay_private(self):
        prompts=([3,5]*66,[8,2]*35);forced={2:240,3:241}
        for reverse in (False,True):
            runner=make_runner(mode='second_reject',draft_tokens=2)
            sessions=[runner.start(prompt,15,{255},64) for prompt in prompts]
            order=[1,0] if reverse else [0,1]
            while any(session.phase=='prefill' for session in sessions):
                for index in order:
                    if sessions[index].phase=='prefill':
                        runner.prefill(sessions[index]);self.assert_canonical(sessions[index],prompts[index])
            while any(not session.done for session in sessions):
                for index in order:
                    if sessions[index].done:continue
                    step(runner,sessions[index],forced if index==0 else {})
                    for current,prompt in zip(sessions,prompts):self.assert_canonical(current,prompt)
            self.assertEqual(sessions[0].generated,autoregressive_oracle(prompts[0],15,forced))
            self.assertEqual(sessions[1].generated,autoregressive_oracle(prompts[1],15))
            self.assertEqual(sessions[1].stats['forced_control_tokens'],0)

    def test_cancel_after_forcing_releases_slot_cache_and_replacement_has_canonical_draft(self):
        runner=make_runner(ContinuousMTPPrefixCache(65536),mode='accept',draft_tokens=2)
        prompt=[3,5]*66;cancelled=[False]
        a=ready(runner,prompt,18,should_cancel=lambda:cancelled[0])
        b=ready(runner,[8,2],12)
        runner.force_next(a,240);self.assert_canonical(a,prompt);self.assert_canonical(b,[8,2])
        cancelled[0]=True;before=len(runner.engine.calls)
        with self.assertRaises(GenerationCancelled):runner.force_next(a,241)
        self.assertEqual(len(runner.engine.calls),before)
        runner.close(a);self.assertIsNone(runner.cache.owner);self.assertIsNone(a.draft_state)
        c=ready(runner,prompt,14);self.assertGreater(c.reused,0)
        while not b.done or not c.done:
            for session,current_prompt in ((c,prompt),(b,[8,2])):
                if not session.done:
                    runner.decode([session]);self.assert_canonical(session,current_prompt)
        self.assertEqual(b.generated,autoregressive_oracle([8,2],12))
        self.assertEqual(c.generated,autoregressive_oracle(prompt,14))
        self.assertEqual(c.stats['forced_control_tokens'],0)

    def test_output_budget_wins_even_when_closing_marker_cannot_finish(self):
        prompt=[3,5,8];runner=make_runner(draft_tokens=2)
        session=ready(runner,prompt,3)
        runner.decode([session],emission_limit=1);runner.force_next(session,240)
        self.assertTrue(session.done);self.assertEqual(session.result(0)['finish_reason'],'length')
        self.assertEqual(session.generated,autoregressive_oracle(prompt,3,{2:240}))
        self.assert_canonical(session,prompt)
        with self.assertRaises(ValueError):runner.force_next(session,241)

    def test_invalid_forcing_and_emission_limits_do_not_change_committed_history(self):
        runner=make_runner();session=ready(runner,[3,5,8],10);before=list(session.generated)
        for token in (255,True,-1):
            with self.subTest(token=token),self.assertRaises(ValueError):runner.force_next(session,token)
        for limit in (0,-1,True,1.5):
            with self.subTest(limit=limit),self.assertRaises(ValueError):runner.decode([session],emission_limit=limit)
        self.assertEqual(session.generated,before)


class DecodeTokenizer:
    def __init__(self,prompt,closing_ids,pieces,starts=True):
        self.prompt=list(prompt);self.closing_ids=list(closing_ids);self.starts=starts
        assert 242 not in pieces and 242 not in self.closing_ids
        self.opening_ids=[242];self.pieces={242:'<think>',**pieces}

    def encode(self,text,**kwargs):
        assert kwargs==dict(add_special_tokens=False)
        if text=='</think>':return list(self.closing_ids)
        assert text=='<think>'
        return list(self.opening_ids)

    def decode(self,ids,**kwargs):
        if list(ids)==self.prompt:
            return '<|im_start|>assistant\n'+('<think>\n' if self.starts else '')
        return ''.join(self.pieces[token] for token in ids)


class ReasoningMTPApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.env=patch.dict(os.environ,SPARK_API_AUDIT=str(self.path/'audit.jsonl'),SPARK_MTP_POLICY='acceptance')
        self.env.start();self.runtime=Runtime(mode='continuous_mtp');self.runtime.ready=True
        self.runtime.batch_runner=make_runner(self.runtime.prefix_cache,draft_tokens=2)
        self.key='fixture-'+32*'r';self.headers={'Authorization':'Bearer '+self.key};self.prompt=[3,5,8]
        self.tokenize=patch('api_server.chat_token_ids',return_value=list(self.prompt));self.tokenize.start()

    def tearDown(self):
        self.runtime.close();self.tokenize.stop();self.env.stop();self.temp.cleanup()

    def configure(self,natural=False,disabled=False,starts=True):
        if disabled:
            sequence=autoregressive_oracle(self.prompt,2);closing=[];pieces={sequence[0]:'42'};eos=sequence[1]
        elif natural:
            sequence=autoregressive_oracle(self.prompt,6);closing=sequence[2:4]
            pieces=dict(zip(sequence[:5],['核','算','</thi','nk>','42']));eos=sequence[5]
        else:
            closing=[240,241];sequence=autoregressive_oracle(self.prompt,6,{2:240,3:241})
            pieces=dict(zip(sequence[:5],['核','算','</thi','nk>','42']));eos=sequence[5]
        self.assertEqual(len(set(sequence)),len(sequence),'Fixture must not alias EOS or text token IDs')
        self.runtime.eos={eos};self.runtime.tokenizer=DecodeTokenizer(self.prompt,closing,pieces,starts=starts and not disabled)
        return sequence

    def body(self,**options):
        return {**dict(model=MODEL,messages=[dict(role='user',content='fixture')],max_tokens=16),**options}

    def rows(self):return [json.loads(line) for line in self.runtime.audit.read_text().splitlines()]

    def test_json_and_sse_forced_budget_have_no_leak_and_exact_usage_audit(self):
        expected=self.configure()
        with TestClient(create_app(self.runtime,self.key)) as client:
            for streaming in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(
                    enable_thinking=True,reasoning_effort='low',thinking_budget=2,stream=streaming))
                self.assertEqual(response.status_code,200,response.text)
                if streaming:
                    events=[line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
                    self.assertEqual(events[-1],'[DONE]');chunks=[json.loads(value) for value in events[:-1]]
                    self.assertEqual(''.join(choice['delta'].get('reasoning_content','') for chunk in chunks for choice in chunk['choices']),'核算')
                    self.assertEqual(''.join(choice['delta'].get('content','') for chunk in chunks for choice in chunk['choices']),'42')
                    usage=next(chunk['usage'] for chunk in chunks if chunk.get('usage'))
                else:
                    data=response.json();message=data['choices'][0]['message']
                    self.assertEqual(message,dict(role='assistant',content='42',reasoning_content='核算'))
                    usage=data['usage']
                self.assertEqual(usage['completion_tokens'],len(expected))
                self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'],2)
                audit=self.rows()[-1];self.assertEqual(audit['status'],'completed')
                self.assertEqual(audit['reasoning']['used_tokens'],2)
                self.assertTrue(audit['reasoning']['budget_triggered'])
                self.assertEqual(audit['reasoning']['forced_control_tokens'],2)
                self.assertEqual(audit['speculative']['forced_control_tokens'],2)

    def test_natural_close_and_unbounded_thinking_are_separate_from_forced_budget(self):
        self.configure(natural=True)
        with TestClient(create_app(self.runtime,self.key)) as client:
            for options in (dict(enable_thinking=True),dict(enable_thinking=True,thinking_budget=8)):
                response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(**options))
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['choices'][0]['message'],dict(role='assistant',content='42',reasoning_content='核算'))
                self.assertEqual(response.json()['usage']['completion_tokens_details']['reasoning_tokens'],4)
                audit=self.rows()[-1];self.assertEqual(audit['reasoning']['forced_control_tokens'],0)
                self.assertFalse(audit['reasoning']['budget_triggered'])

    def test_disabled_thinking_ordinary_answer_has_no_reasoning_field(self):
        self.configure(disabled=True)
        with TestClient(create_app(self.runtime,self.key)) as client:
            response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(reasoning_effort='none'))
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['choices'][0]['message'],dict(role='assistant',content='42'))
            self.assertNotIn('completion_tokens_details',response.json()['usage'])

    def test_length_mid_forced_multitoken_close_does_not_leak_half_tag_or_exceed_budget(self):
        self.configure()
        with TestClient(create_app(self.runtime,self.key)) as client:
            for streaming in (False,True):
                response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(
                    enable_thinking=True,thinking_budget=2,max_tokens=3,stream=streaming))
                self.assertEqual(response.status_code,200,response.text)
                if streaming:
                    events=[line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
                    self.assertEqual(events[-1],'[DONE]')
                    chunks=[json.loads(value) for value in events[:-1]]
                    choices=[choice for chunk in chunks for choice in chunk['choices']]
                    self.assertEqual(''.join(choice['delta'].get('reasoning_content','') for choice in choices),'核算')
                    self.assertEqual(''.join(choice['delta'].get('content','') for choice in choices),'')
                    self.assertEqual([choice['finish_reason'] for choice in choices if choice.get('finish_reason')],['length'])
                    usage=next(chunk['usage'] for chunk in chunks if chunk.get('usage'))
                else:
                    data=response.json();self.assertEqual(data['choices'][0]['finish_reason'],'length')
                    self.assertEqual(data['choices'][0]['message'],dict(role='assistant',content='',reasoning_content='核算'))
                    usage=data['usage']
                self.assertEqual(usage['completion_tokens'],3)
                self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'],2)
                audit=self.rows()[-1];self.assertEqual(audit['status'],'completed')
                self.assertEqual(audit['reasoning']['used_tokens'],2)
                self.assertEqual(audit['reasoning']['forced_control_tokens'],1)

    def test_cancel_between_committed_control_step_and_framing_accounts_final_step(self):
        self.configure();original=ContinuousRequest.after_step;cancelled=[]
        def after_step(request):
            if len(request.generation.generated)==3 and not cancelled:
                cancelled.append(True);self.runtime.cancel_job(request.job)
            return original(request)
        options=normalize_request(self.body(enable_thinking=True,thinking_budget=2),MODEL,self.runtime.max_output)
        with patch.object(ContinuousRequest,'after_step',after_step):
            job=self.runtime.submit(options)
            self.assertIsNone(job.future.result(3))
        self.assertEqual(cancelled,[True]);self.assertEqual(self.runtime.pending,0)
        audit=self.rows()[-1];self.assertEqual(audit['id'],job.id)
        self.assertEqual(audit['status'],'cancelled');self.assertEqual(audit['output_tokens'],3)
        self.assertEqual(audit['reasoning']['used_tokens'],2)
        self.assertTrue(audit['reasoning']['budget_triggered'])
        self.assertEqual(audit['reasoning']['forced_control_tokens'],1)
        self.assertEqual(job.options,{})

    def test_budget_uses_actual_prompt_boundary_and_legacy_modes_reject_finite_budget(self):
        self.configure(starts=False)
        with TestClient(create_app(self.runtime,self.key)) as client:
            response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(enable_thinking=True,thinking_budget=2))
            self.assertEqual(response.status_code,400,response.text)
        for mode in ('serial_mtp','continuous_greedy'):
            runtime=Runtime(mode=mode);runtime.ready=True
            try:
                with patch.object(runtime,'submit') as submit,TestClient(create_app(runtime,self.key)) as client:
                    for streaming in (False,True):
                        response=client.post('/v1/chat/completions',headers=self.headers,json=self.body(reasoning_effort='low',stream=streaming))
                        self.assertEqual(response.status_code,400,response.text)
                    submit.assert_not_called()
            finally:runtime.close()


if __name__=='__main__':unittest.main()
