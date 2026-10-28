"""Authenticated GB10 API with serial, greedy batch and cooperative MTP modes."""
import asyncio
from concurrent.futures import Future,CancelledError as FutureCancelledError
from dataclasses import dataclass
import hmac
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import torch
from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,StreamingResponse

from api_protocol import normalize_request,OutputFramer,RequestError
from tool_schema import ToolSchemaError,ToolCallError
from reasoning_control import ReasoningControl
from output_decoder import IncrementalOutputDecoder
from serial_executor import SerialExecutor
from continuous_scheduler import ContinuousScheduler,SchedulerFull
from context_pool import ContextPool
from preparation_executor import PreparationExecutor,PreparationFull
from engine.speculative import generate_mtp
from engine.request_control import GenerationCancelled
from engine.prefix_cache import PrefixCache
from engine.tokenization import chat_token_ids,generation_eos_ids,prompt_starts_in_reasoning

MODEL=os.environ.get('SPARK_MODEL_ID','qwen3.8-flash-next-gb10')
# Legacy public constant for older callers/tests. Admission and metadata use
# the Runtime instance so an experimental mode cannot change another service.
MAX_CONTEXT=int(os.environ.get('SPARK_MAX_CONTEXT','32768'))
CONTINUOUS_MTP_CONTEXT=262144
# Compatibility fallback for runtimes without per-instance settings.
MAX_OUTPUT=int(os.environ.get('SPARK_MAX_OUTPUT','4096'))
PREFILL_CHUNK=int(os.environ.get('SPARK_PREFILL_CHUNK','2048'))
LOW_SYNC=os.environ.get('SPARK_LOW_SYNC','0')=='1'
BORROWED_CACHE=os.environ.get('SPARK_BORROWED_CACHE','0')=='1'
QSA_PREFILL_MODE=os.environ.get('SPARK_QSA_PREFILL_MODE','predecoded')
# MTP draft routed-expert top-k; only changes draft acceptance (the target verifies every token).
DRAFT_TOP_K=int(os.environ.get('SPARK_DRAFT_TOPK','5'))
LOG=logging.getLogger('spark_api')
REASONING_EFFORTS=dict(none=0,low=1024,medium=4096,high=16384)


def public_error(error):
    if isinstance(error,(RequestError,ToolSchemaError,ToolCallError)):
        return dict(message=str(error),
            type='server_error' if isinstance(error,ToolCallError) else 'invalid_request_error',
            code=getattr(error,'code','invalid_request'),param=getattr(error,'param',None),
            **({'tool':error.tool} if getattr(error,'tool',None) else {}))
    return dict(message='The model request failed. Retry or reduce the output budget.',
                type='server_error',code='model_request_failed',param=None)


def record_failure(job,error):
    job.start_error=error
    value=public_error(error)
    # Known tool errors contain only bounded names/paths. Unexpected exception
    # text/tracebacks can contain model arguments, so log the class only.
    LOG.error('Request %s failed: %s code=%s tool=%s param=%s',job.id,
              type(error).__name__,value['code'],value.get('tool'),value.get('param'))
    job.events.put(('error',value))
    return {k:value[k] for k in ('code','tool','param') if k in value}


def request_http_error(error):
    return HTTPException(400 if isinstance(error,(RequestError,ToolSchemaError)) else 502,
                         detail=public_error(error))


def request_token_ids(tokenizer, options):
    extra = {} if options.get('preserve_thinking') is None else dict(preserve_thinking=options['preserve_thinking'])
    return chat_token_ids(tokenizer,options['messages'],options['thinking'],options['tools'] or None,**extra)


def request_reasoning(tokenizer, ids, options, mode, eos):
    starts=prompt_starts_in_reasoning(tokenizer,ids)
    budget=options.get('thinking_budget')
    if budget is not None and (mode!='continuous_mtp' or not starts):
        raise RequestError('A thinking budget requires continuous_mtp and an open checkpoint think prompt.')
    closing=list(map(int,tokenizer.encode('</think>',add_special_tokens=False))) if starts else []
    opening=list(map(int,tokenizer.encode('<think>',add_special_tokens=False))) if starts else []
    if any(token in eos for token in closing):
        raise RequestError('The checkpoint closing think marker must not be EOS.')
    return starts,ReasoningControl(starts,options.get('reasoning_effort'),budget,closing,
                                  opening_ids=opening)


QUEUE_TIMEOUT=float(os.environ.get('SPARK_QUEUE_TIMEOUT_SECONDS','120'))
if not 0 < QUEUE_TIMEOUT <= 600:raise ValueError('Queue timeout must be in (0, 600] seconds')


class QueueDeadlineExceeded(Exception):
    pass


@dataclass
class PreparedRequest:
    ids: list
    output_budget: int
    tokenization_seconds: float


class Job:
    def __init__(self,options,queue_timeout=QUEUE_TIMEOUT):
        self.options=options; self.id='chatcmpl-'+uuid.uuid4().hex
        self.created=int(time.time()); self.received=time.perf_counter(); self.started=threading.Event()
        self.cancel=threading.Event(); self.events=queue.Queue()
        self.start_error=None; self.input_tokens=0; self.future=None
        self.executing=threading.Event(); self.released=False
        self.queue_deadline=self.received+queue_timeout
        self.cancel_reason=None; self.cancel_requested=None
        self.phase='queued'; self.completed_prefill_tokens=0; self.checkpoint_tokens=0
        self.preparation=None


DRAFT_VALUE=os.environ.get("SPARK_DRAFT_TOKENS","adaptive")
DRAFT_TOKENS="adaptive" if DRAFT_VALUE=="adaptive" else int(DRAFT_VALUE)
if DRAFT_TOKENS not in (1,2,"adaptive"):raise ValueError("Invalid MTP draft depth")


class Runtime:
    def __init__(self,mode=None,max_active=None,max_queued=None,total_context=None,prepare_workers=None,mtp_batch=None):
        self.mode=mode or os.environ.get('SPARK_EXECUTION_MODE','serial_mtp')
        if self.mode not in ('serial_mtp','continuous_greedy','continuous_mtp'):raise ValueError('Unknown execution mode')
        self.continuous=self.mode!='serial_mtp'
        self.mtp_enabled=self.mode!='continuous_greedy'
        batch_flag=os.environ.get('SPARK_MTP_BATCH','0')
        if mtp_batch is None:
            if batch_flag not in ('0','1'):raise ValueError('SPARK_MTP_BATCH must be 0 or 1')
            mtp_batch=batch_flag=='1'
        if type(mtp_batch) is not bool or mtp_batch and self.mode!='continuous_mtp':
            raise ValueError('MTP batching requires continuous_mtp mode')
        self.mtp_batch=mtp_batch
        default_context=CONTINUOUS_MTP_CONTEXT if self.mode=='continuous_mtp' else 32768
        self.max_context=int(os.environ.get('SPARK_MAX_CONTEXT',str(default_context)))
        if self.max_context<=0:raise ValueError('Context limit must be a positive integer')
        default_output=0 if self.mode=='continuous_mtp' else 4096
        self.max_output=int(os.environ.get('SPARK_MAX_OUTPUT',str(default_output)))
        if self.max_output<0:raise ValueError('Output limit must be zero or a positive integer')
        self.output_budget_policy='remaining_context' if self.max_output==0 else 'fixed_limit'
        self.mtp_policy=os.environ.get('SPARK_MTP_POLICY','acceptance')
        if self.mtp_policy not in ('acceptance','cost'):raise ValueError('Unknown MTP policy')
        if self.mtp_policy=='cost' and (self.mode!='continuous_mtp' or DRAFT_TOKENS!='adaptive'):
            raise ValueError('Cost policy requires continuous_mtp with adaptive draft depth')
        if self.mtp_batch and self.mtp_policy!='acceptance':raise ValueError('MTP batching requires acceptance policy')
        self.max_active=int(max_active if max_active is not None else os.environ.get('SPARK_MAX_ACTIVE','2' if self.continuous else '1'))
        self.max_queued=int(max_queued if max_queued is not None else os.environ.get('SPARK_MAX_QUEUED','1'))
        active_limit=4 if self.mode=='continuous_mtp' else 2
        if not 1<=self.max_active<=active_limit or not 0<=self.max_queued<=32:raise ValueError('Invalid active/queued capacity')
        if self.mode=='serial_mtp' and self.max_active!=1:raise ValueError('Serial MTP requires one active slot')
        if self.continuous and PREFILL_CHUNK!=2048:
            raise ValueError('Continuous modes preserve the accepted 2048-token prefill boundary')
        self.total_context=int(total_context if total_context is not None else os.environ.get(
            'SPARK_TOTAL_CONTEXT',str(self.max_active*self.max_context)))
        self.context_pool=ContextPool(self.total_context,self.max_context)
        if not self.continuous and self.total_context!=self.max_context:
            raise ValueError('Shared context admission requires continuous mode')
        self.prepare_workers=int(prepare_workers if prepare_workers is not None else os.environ.get(
            'SPARK_PREPARE_WORKERS','2' if self.max_active>2 else '0'))
        if not 0<=self.prepare_workers<=2:raise ValueError('Expected 0..2 CPU preparation workers')
        if self.prepare_workers and not self.continuous:raise ValueError('CPU preparation requires continuous mode')
        # Shared-pool admission needs a known prompt length before GPU checkout.
        self.pooled_admission=self.continuous and (self.prepare_workers>0 or
            self.total_context!=self.max_active*self.max_context)
        if self.pooled_admission and not self.prepare_workers:self.prepare_workers=1
        if self.prepare_workers>self.max_active+self.max_queued:
            raise ValueError('CPU preparation workers exceed pending request capacity')
        self.preparer=(PreparationExecutor(self.prepare_workers,self.max_active+self.max_queued)
            if self.prepare_workers else None)
        self.executor=SerialExecutor()
        self.lock=threading.Lock(); self.audit_lock=threading.Lock()
        self.pending=0; self.ready=False; self.fatal=False; self.queue_timeout=QUEUE_TIMEOUT
        self.audit=Path(os.environ.get('SPARK_API_AUDIT','/workspace/logs/api/requests.jsonl'))
        self.audit.parent.mkdir(parents=True,exist_ok=True)
        if self.mode=='continuous_greedy':
            from engine.greedy_prefix_cache import GreedyPrefixCache
            self.prefix_cache=GreedyPrefixCache(max_bytes=1024**3)
        elif self.mode=='continuous_mtp':
            from engine.continuous_mtp import ContinuousMTPPrefixCache
            self.prefix_cache=ContinuousMTPPrefixCache(max_bytes=1024**3)
        else:
            self.prefix_cache=PrefixCache(max_bytes=1024**3)
        if self.continuous:
            self.scheduler=ContinuousScheduler(self.executor,lambda job:ContinuousRequest(self,job),
                self._decode_ready_batch,
                max_active=self.max_active,max_queued=self.max_queued,fatal=lambda:self.fatal,
                on_start_error=self._failed_admission,decode_together=self.mode=='continuous_greedy',
                prepare=self._submit_preparation if self.pooled_admission else None,
                admit=self._admit_prepared if self.pooled_admission else None,
                discard=self._discard_preparation if self.pooled_admission else None,mtp_batch=self.mtp_batch)
        else:
            self.scheduler=None

    def _decode_ready_batch(self,sessions):
        if not self.mtp_batch:
            return self.batch_runner.decode([s.generation for s in sessions])
        errors=[None]*len(sessions);indices=[];limits=[];forced=[]
        for index,session in enumerate(sessions):
            try:
                token,limit=session.reasoning.plan()
                indices.append(index);forced.append(token);limits.append(limit)
            except Exception as error:
                errors[index]=error
        if indices:
            try:
                results=self.batch_runner.decode_batch([sessions[i].generation for i in indices],
                    emission_limits=limits,forced_tokens=forced)
                if len(results)!=len(indices):raise RuntimeError('MTP batch lane result count differs')
            except Exception as error:
                results=[error]*len(indices)
            for index,error in zip(indices,results):errors[index]=error
        return errors

    def validate_context_config(self,config):
        maximum=config.get('max_position_embeddings') if isinstance(config,dict) else None
        if type(maximum) is not int or maximum<=0:
            raise ValueError('Checkpoint must declare a positive max_position_embeddings')
        if self.max_context>maximum:
            raise ValueError('API context exceeds checkpoint configuration.')

    def validate_request_context(self,ids,budget):
        # The per-request window and shared admission pool are independent.
        if len(ids)+budget>self.max_context:
            raise RequestError(f'Prompt plus output budget exceeds the per-request context of {self.max_context} tokens.')

    def resolve_output_budget(self,ids,budget):
        if budget is None and self.max_output==0:
            budget=self.max_context-len(ids)
        if type(budget) is not int or budget<1 or (self.max_output and budget>self.max_output):
            limit=f'1..{self.max_output}' if self.max_output else 'a positive integer within the remaining context'
            raise RequestError(f'max_tokens must be {limit}.')
        self.validate_request_context(ids,budget)
        return budget

    def _submit_preparation(self,job):
        # Only bounded admitted+queued jobs reach this executor. These workers
        # tokenize CPU data; model state and CUDA remain on SerialExecutor.
        options=job.options
        def prepare():
            if job.cancel.is_set():raise GenerationCancelled('queued')
            begin=time.perf_counter()
            ids=request_token_ids(self.tokenizer,options)
            seconds=time.perf_counter()-begin
            if job.cancel.is_set():raise GenerationCancelled('tokenization')
            budget=self.resolve_output_budget(ids,options['max_tokens'])
            try:self.context_pool.validate(len(ids),budget)
            except ValueError as error:raise RequestError(str(error)) from error
            return PreparedRequest(ids,budget,seconds)
        job.preparation=self.preparer.submit(prepare)
        return job.preparation

    def _admit_prepared(self,job):
        if job.cancel.is_set():raise GenerationCancelled('queued')
        prepared=job.preparation.result()
        return self.context_pool.try_reserve(job.id,len(prepared.ids),prepared.output_budget)

    def _discard_preparation(self,job):
        if job is None:return
        if job.preparation is not None:job.preparation.cancel()
        job.preparation=None
        self.context_pool.release(job.id)

    def append_audit(self,value):
        with self.audit_lock:
            with self.audit.open('a') as file:file.write(json.dumps(value,ensure_ascii=False)+'\n')

    def generate(self,ids,budget,callback=None,cache=True,should_cancel=None,on_progress=None):
        budget=self.resolve_output_budget(ids,budget)
        if self.continuous:
            extra=dict(should_cancel=should_cancel) if self.mode=='continuous_mtp' else {}
            session=self.batch_runner.start(ids,budget,self.eos,PREFILL_CHUNK,
                callback,on_progress,cache=cache,**extra)
            emitted=0
            try:
                while not session.done:
                    if should_cancel and should_cancel():raise GenerationCancelled(session.phase,session.position,session.checkpoint_tokens)
                    if session.phase=='prefill':self.batch_runner.prefill(session)
                    else:self.batch_runner.decode([session])
                    session.progress()
                    if len(session.generated)!=emitted:
                        session.emit();emitted=len(session.generated)
                return session.result(self.prefix_cache.bytes),session.state
            finally:self.batch_runner.close(session)
        return generate_mtp(self.engine,self.draft,ids,budget,self.eos,PREFILL_CHUNK,
            low_sync=LOW_SYNC,borrowed_graph_cache=BORROWED_CACHE,on_tokens=callback,
            prefix_cache=self.prefix_cache if cache else None,draft_tokens=DRAFT_TOKENS,
            should_cancel=should_cancel,on_progress=on_progress)

    @torch.no_grad()
    def load(self):
        from transformers import AutoTokenizer
        from engine.model import GreedyEngineRef
        from engine.mtp import MTPDraft
        from engine.weights import WeightStore
        torch.set_num_threads(4); torch.cuda.set_device(0)
        begin=time.perf_counter()
        model_dir=os.environ.get('SPARK_MODEL_DIR','/model')
        self.cfg=json.loads((Path(model_dir)/'config.json').read_text())['text_config']
        self.validate_context_config(self.cfg)
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        self.eos=generation_eos_ids(self.tokenizer,self.cfg,model_dir)
        strategy=os.environ.get('SPARK_MOE_STRATEGY','heuristic')
        self.moe_strategy=None
        if strategy=='deterministic_v1':
            from engine.ops.moe_strategy_domain import DeterministicMoEDomain
            self.moe_strategy=DeterministicMoEDomain(Path(__file__).parent/'engine/ops/moe_deterministic_sm121.json')
            self.moe_strategy.enabled=True
        elif strategy=='tuned_top10_v1':
            from engine.ops.moe_strategy_domain import TunedTop10MoEDomain
            self.moe_strategy=TunedTop10MoEDomain()
        elif strategy!='heuristic':raise ValueError('Unknown deterministic MoE strategy')
        store=WeightStore(model_dir,'cuda')
        self.engine=GreedyEngineRef(store,self.cfg,activation_backend='sm121-native',router_backend='native',
            moe_finalize='deterministic',expert_backend='flashinfer',head_backend='sm121',qsa_backend='triton',
            resident_experts=True,graph_decode=True,graph_verify=self.mtp_enabled,
            hc_projection_backend='sm121-two',kv_backend='auto')
        self.gdn_nvfp4=dict(enabled=False)
        nvfp4_dir=os.environ.get('SPARK_GDN_NVFP4_DIR','')
        if nvfp4_dir:
            # Offline GPTQ+low-rank NVFP4 W4A16 GDN qkvz/out_proj weights; every file is
            # hash-checked against the manifest before use.
            import hashlib
            from engine.ops import nvfp4_linear
            manifest_path=Path(nvfp4_dir)/'manifest.json'
            expected=os.environ.get('SPARK_GDN_NVFP4_MANIFEST_SHA256','')
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=expected:
                raise ValueError('NVFP4 GDN manifest hash differs')
            manifest=json.loads(manifest_path.read_text())
            for name,digest in manifest['files'].items():
                with (Path(nvfp4_dir)/name).open('rb') as handle:
                    if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:
                        raise ValueError('NVFP4 GDN weight file differs: '+name)
            attached=nvfp4_linear.attach(self.engine,nvfp4_dir)
            nvfp4_linear.ACTIVE.update({'qkvz','out'})
            self.gdn_nvfp4=dict(enabled=True,projections=['qkvz','out'],attached=attached,
                                manifest_sha256=expected,method=manifest.get('method'))
        draft_head=self.engine.logits
        self.draft_head_metadata=dict(enabled=False,full_vocab_size=int(self.engine.head_qweight.shape[1]))
        vocab_path=os.environ.get('SPARK_DRAFT_VOCAB_PATH','')
        vocab_sha=os.environ.get('SPARK_DRAFT_VOCAB_SHA256','')
        if bool(vocab_path)!=bool(vocab_sha):
            raise ValueError('Draft vocabulary path and SHA256 must be supplied together')
        if vocab_path:
            if not self.mtp_enabled:raise ValueError('A compact draft head requires MTP')
            import hashlib
            from engine.draft_head import DraftVocabularyHead,tokenizer_vocabulary_requirements
            defined,mandatory=tokenizer_vocabulary_requirements(self.tokenizer,self.eos)
            tokenizer_sha=hashlib.sha256((Path(model_dir)/'tokenizer.json').read_bytes()).hexdigest()
            draft_head=DraftVocabularyHead.from_manifest(self.engine.head_qweight,self.engine.head_scales,
                vocab_path,expected_manifest_sha256=vocab_sha,expected_tokenizer_sha256=tokenizer_sha,
                mandatory_ids=mandatory,defined_ids=defined)
            self.draft_head_metadata=dict(enabled=True,**draft_head.metadata)
        deep_head=None
        deep_path=os.environ.get('SPARK_DRAFT_DEEP_VOCAB_PATH',''); deep_sha=os.environ.get('SPARK_DRAFT_DEEP_VOCAB_SHA256','')
        if bool(deep_path)!=bool(deep_sha):
            raise ValueError('Deep draft vocabulary path and SHA256 must be supplied together')
        if deep_path:
            # Draft-only smaller head for recursive steps beyond the first proposal (single-lane deep drafting).
            import hashlib
            from engine.draft_head import DraftVocabularyHead,tokenizer_vocabulary_requirements
            defined,mandatory=tokenizer_vocabulary_requirements(self.tokenizer,self.eos)
            tokenizer_sha=hashlib.sha256((Path(model_dir)/'tokenizer.json').read_bytes()).hexdigest()
            deep_head=DraftVocabularyHead.from_manifest(self.engine.head_qweight,self.engine.head_scales,
                deep_path,expected_manifest_sha256=deep_sha,expected_tokenizer_sha256=tokenizer_sha,
                mandatory_ids=mandatory,defined_ids=defined)
            self.draft_head_metadata['deep_head']=deep_head.metadata
        self.draft=(MTPDraft(store,self.cfg,self.engine.embedding,draft_head,top_k=DRAFT_TOP_K,kv_backend='auto')
                    if self.mtp_enabled else None)
        if self.draft is not None:self.draft.deep_head=deep_head
        if self.mode=='continuous_greedy':
            from engine.continuous import GreedyBatchRunner
            self.batch_runner=GreedyBatchRunner(self.engine,self.prefix_cache)
        elif self.mode=='continuous_mtp':
            from engine.continuous_mtp import ContinuousMTPRunner
            if self.mtp_batch:
                from engine.batched_mtp import BatchedMTPRunner
                ContinuousMTPRunner=BatchedMTPRunner
            from engine.speculative_policy import CostAwareDraftDepth
            self.batch_runner=ContinuousMTPRunner(self.engine,self.draft,self.prefix_cache,
                draft_tokens=DRAFT_TOKENS,
                policy_factory=CostAwareDraftDepth if self.mtp_policy=='cost' else None)
        if QSA_PREFILL_MODE not in ('packed','predecoded'):raise ValueError('Unknown SM121 QSA prefill mode.')
        for layer in self.engine.layers:
            if layer.layer_type=='full_attention':layer.attention.prefill_mode=QSA_PREFILL_MODE
        if self.draft is not None:self.draft.attention.prefill_mode=QSA_PREFILL_MODE
        torch.cuda.synchronize(); load_seconds=time.perf_counter()-begin
        ids=chat_token_ids(self.tokenizer,[dict(role='user',content='请只输出这个JSON，不要解释：{"answer":42,"ok":true}')],False)
        rows=[]
        for repeat in range(2):
            observed=[]
            callback=(lambda tokens:observed.append(tuple(tokens))) if repeat else None
            row,state=self.generate(ids,64,callback,cache=False)
            raw=self.tokenizer.decode(row['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if row['finish_reason']!='stop' or json.loads(raw.strip())!=dict(answer=42,ok=True):
                raise RuntimeError('Real warmup JSON answer failed.')
            if callback and observed[-1]!=tuple(row['generated_ids']):raise RuntimeError('Streaming callback dropped tokens.')
            row['visible_text']=raw; rows.append(row); del state
        if rows[0]['generated_ids']!=rows[1]['generated_ids']:raise RuntimeError('Streaming changed model output.')
        long_warmup=[]
        length=int(os.environ.get('SPARK_LONG_WARMUP_TOKENS','0'))
        if length:
            from engine.benchmark_long_context import case_for_length
            case=case_for_length(self.tokenizer,length)
            row,state=self.generate(case['input_ids'],96,cache=False)
            raw=self.tokenizer.decode(row['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if row['finish_reason']!='stop' or json.loads(raw.split('</think>')[-1].strip())!=case['expected']:
                raise RuntimeError('Long-context real warmup answer failed.')
            row.update(input_tokens=len(case['input_ids']),visible_text=raw)
            long_warmup.append(row);del state
        imports=[name for name in __import__('sys').modules if name=='vllm' or name.startswith('vllm.')]
        if imports:raise RuntimeError('Independent runtime imported vLLM Python.')
        status=dict(model=MODEL,load_seconds=load_seconds,max_context=self.max_context,max_output=self.max_output,
                    max_output_tokens=self.max_output or None,output_budget_policy=self.output_budget_policy,
                    context_window_per_request=self.max_context,
                    aggregate_context_budget=self.total_context,
                    context_pool=self.context_pool.snapshot(),prepare_workers=self.prepare_workers,
                    warmup=rows,long_warmup=long_warmup,prefill_chunk_size=PREFILL_CHUNK,
                    low_sync=LOW_SYNC if self.mode=='serial_mtp' else True,
                    borrowed_graph_cache=BORROWED_CACHE if self.mode=='serial_mtp' else False,
                    qsa_prefill_mode=QSA_PREFILL_MODE,draft_tokens=DRAFT_TOKENS if self.mtp_enabled else None,
                    draft_head=self.draft_head_metadata,gdn_nvfp4=self.gdn_nvfp4,
                    exact_batch_optins={k:os.environ.get(k,'0') for k in ('SPARK_HC_FUSED','SPARK_HEAD_ROWS','SPARK_MTP_PAD','SPARK_DRAFT_BATCH','SPARK_LAZY_CLONE')},
                    queue_timeout_seconds=self.queue_timeout,max_pending_requests=self.max_active+self.max_queued,
                    execution_mode=self.mode,max_active_requests=self.max_active,max_queued_requests=self.max_queued,
                    mtp_enabled=self.mtp_enabled,owned_graph_cache=self.continuous,
                    scheduling='decode_priority_batch' if self.mtp_batch else 'cooperative_round_robin' if self.mode=='continuous_mtp' else self.mode,
                    batched_verification=self.mtp_batch,mtp_policy=self.mtp_policy if self.mtp_enabled else None,
                    batch_execution=self.batch_runner.snapshot() if self.mtp_batch else None,
                    experimental=self.mode=='continuous_mtp',
                    moe_strategy=strategy,actual_moe_strategy_counts=self.moe_strategy.observed if self.moe_strategy else {},
                    prefix_cache_max_bytes=self.prefix_cache.max_bytes,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    runtime_vllm_modules=imports,ready_at=int(time.time()))
        (self.audit.parent/'startup.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        self.ready=True
        print('ENGINE_READY',MODEL,'load_seconds',round(load_seconds,3),flush=True)

    def _release(self,job):
        with self.lock:
            if job.released:return False
            job.released=True
            self.pending-=1
            self._discard_preparation(job)
            return True

    def cancel_job(self,job,reason='client_disconnected'):
        with self.lock:
            if job.released:return False
            if reason=='queue_timeout' and (job.executing.is_set() or time.perf_counter()<job.queue_deadline):
                return False
            if job.cancel_requested is None:
                job.cancel_requested=time.perf_counter();job.cancel_reason=reason
            if job.cancel_reason=='queue_timeout':job.start_error=QueueDeadlineExceeded('The queue wait deadline was exceeded. Retry shortly.')
            job.cancel.set()
            future=job.future
        # Future callbacks must run outside Runtime.lock. A cancelled queued
        # future never enters run(), so its completion callback owns release.
        if future is not None:future.cancel()
        return True

    def expire_queued(self,job):
        return self.cancel_job(job,'queue_timeout')

    def _queued_done(self,job,future):
        if not future.cancelled() or not self._release(job):return
        job.started.set()
        status='queue_timeout' if job.cancel_reason=='queue_timeout' else 'cancelled_queued'
        job.events.put(('cancelled',{}))
        now=time.perf_counter()
        try:
            self.append_audit(dict(id=job.id,created=job.created,status=status,phase='queued',
                input_tokens=0,output_tokens=0,queue_seconds=now-job.received,
                server_seconds=now-job.received,stream=job.options['stream'],
                cancel_latency_seconds=now-job.cancel_requested if job.cancel_requested is not None else None,
                completed_prefill_tokens=0,checkpoint_tokens=0,queue_timeout_seconds=self.queue_timeout))
        except OSError:LOG.exception('Unable to record request %s audit',job.id)
        finally:job.options={}

    def _abort_submission(self,job):
        # Once a scheduler Future is published, only cancellation plus owner
        # cleanup may return a live lease. Queued/pre-enqueue work has no state.
        self.cancel_job(job,'submission_failed')
        if job.future is None or job.future.cancelled():
            self._release(job)
            job.options={}

    def submit(self,options):
        with self.lock:
            if not self.ready or self.fatal:raise HTTPException(503,'Engine is unavailable.')
            if self.pending>=self.max_active+self.max_queued:
                raise HTTPException(429,f'{self.max_active} active and {self.max_queued} queued request slots are full. Retry shortly.',headers={'Retry-After':'1'})
            self.pending+=1
        job=Job(options,self.queue_timeout)
        if self.scheduler:job.future=Future()
        try:
            if self.scheduler:self.scheduler.submit(job,future=job.future)
            else:job.future=self.executor.submit(self.run,job)
            job.future.add_done_callback(lambda future:self._queued_done(job,future))
        except (SchedulerFull,PreparationFull) as error:
            self._abort_submission(job)
            raise HTTPException(429,str(error),headers={'Retry-After':'1'}) from error
        except BaseException:
            self._abort_submission(job)
            raise
        return job

    def close(self):
        if self.scheduler:self.scheduler.shutdown()
        if self.preparer:self.preparer.shutdown(wait=True,cancel_futures=True)
        self.executor.shutdown(wait=True,cancel_futures=True)

    def _failed_admission(self,job,error):
        job.start_error=error;job.started.set()
        job.events.put(('error','The model request could not be started.'))
        try:self.append_audit(dict(id=job.id,status='error',phase='admission',execution_mode=self.mode))
        finally:job.options={};self._release(job)
        raise error

    @torch.no_grad()
    def run(self,job):
        begin=time.perf_counter(); options=job.options; emitted_count=0; first_output_seconds=None
        queue_seconds=begin-job.received;tokenization_seconds=0.;audited=False
        try:
            with self.lock:
                if time.perf_counter()>=job.queue_deadline and not job.cancel.is_set():
                    job.cancel_requested=time.perf_counter();job.cancel_reason='queue_timeout'
                    job.cancel.set();job.start_error=QueueDeadlineExceeded('The queue wait deadline was exceeded. Retry shortly.')
                job.executing.set()
            if job.cancel.is_set():raise GenerationCancelled('queued')
            job.phase='tokenization'
            tokenize_begin=time.perf_counter()
            ids=request_token_ids(self.tokenizer,options)
            tokenization_seconds=time.perf_counter()-tokenize_begin
            job.input_tokens=len(ids)
            if job.cancel.is_set():raise GenerationCancelled('tokenization')
            options['max_tokens']=self.resolve_output_budget(ids,options['max_tokens'])
            job.started.set()
            def delta(value):
                nonlocal first_output_seconds
                if first_output_seconds is None and any(value.get(key) for key in ('content','reasoning_content','tool_calls')):
                    first_output_seconds=time.perf_counter()-job.received
                if options['stream']:job.events.put(('delta',value))
            starts,reasoning=request_reasoning(self.tokenizer,ids,options,self.mode,self.eos)
            framer=OutputFramer(options['tools'],delta,starts_in_reasoning=starts)
            def on_tokens(tokens):
                nonlocal emitted_count
                emitted_count=len(tokens)
                if job.cancel.is_set():raise GenerationCancelled()
                visible=[token for token in tokens if token not in self.eos]
                raw=self.tokenizer.decode(visible,skip_special_tokens=False,clean_up_tokenization_spaces=False).rstrip('\ufffd')
                framer.feed(raw)
                reasoning.observe(tokens)
            def progress(phase,completed,checkpoint):
                job.phase=phase;job.completed_prefill_tokens=completed;job.checkpoint_tokens=checkpoint
            row,state=self.generate(ids,options['max_tokens'],on_tokens,
                should_cancel=job.cancel.is_set,on_progress=progress)
            if job.cancel.is_set():
                del state
                raise GenerationCancelled(job.phase,job.completed_prefill_tokens,job.checkpoint_tokens)
            # The final token can end in a partial UTF-8 sequence when length-limited.
            visible=[token for token in row['generated_ids'] if token not in self.eos]
            framer.feed(self.tokenizer.decode(visible,skip_special_tokens=False,clean_up_tokenization_spaces=False),final=True)
            del state
            finish='tool_calls' if framer.calls else row['finish_reason']
            usage=dict(prompt_tokens=len(ids),completion_tokens=row['output_tokens'],total_tokens=len(ids)+row['output_tokens'],
                prompt_tokens_details=dict(cached_tokens=row['prefix_cache']['reused_tokens']))
            if reasoning.enabled:
                usage['completion_tokens_details']=dict(reasoning_tokens=reasoning.used)
            response=dict(id=job.id,object='chat.completion',created=job.created,model=MODEL,
                choices=[dict(index=0,message=framer.message(),finish_reason=finish)],usage=usage)
            audit=dict(id=job.id,created=job.created,input_tokens=len(ids),output_tokens=row['output_tokens'],
                output_budget=options['max_tokens'],output_budget_policy=self.output_budget_policy,
                finish_reason=finish,first_token_seconds=row['first_token_seconds'],
                decode_tokens_per_second=row['decode_tokens_per_second'],request_seconds=row['request_seconds'],
                speculative=row['speculative'],stream=options['stream'],status='completed',
                queue_seconds=queue_seconds,tokenization_seconds=tokenization_seconds,
                first_output_seconds=first_output_seconds,server_seconds=time.perf_counter()-job.received,
                prefill_chunks=row['prefill_chunks'],target_prefill_seconds=row['target_prefill_seconds'],
                draft_prefill_before_first_seconds=row['draft_prefill_before_first_seconds'],
                draft_prefill_after_first_seconds=row['draft_prefill_after_first_seconds'],
                prefix_cache=row['prefix_cache'],prefill_chunk_size=PREFILL_CHUNK,
                low_sync=LOW_SYNC,borrowed_graph_cache=BORROWED_CACHE,reasoning=reasoning.snapshot())
            self.append_audit(audit);audited=True
            job.events.put(('done',dict(finish_reason=finish,usage=usage)))
            return response
        except GenerationCancelled as error:
            if torch.cuda.is_initialized():torch.cuda.synchronize()
            if error.phase!='request':job.phase=error.phase
            job.completed_prefill_tokens=max(job.completed_prefill_tokens,error.completed_prefill_tokens)
            job.checkpoint_tokens=max(job.checkpoint_tokens,error.checkpoint_tokens)
            job.events.put(('cancelled',{}))
            return None
        except Exception as error:
            job.start_error=error
            if isinstance(error,RuntimeError) and any(term in str(error).lower() for term in ('cuda','nonfinite')):
                self.fatal=True; self.ready=False
            record_failure(job,error)
            raise
        finally:
            if not audited:
                try:
                    self.append_audit(dict(id=job.id,created=job.created,status=('queue_timeout' if job.cancel_reason=='queue_timeout' else 'cancelled' if job.cancel.is_set() else 'error'),
                        phase=job.phase,completed_prefill_tokens=job.completed_prefill_tokens,checkpoint_tokens=job.checkpoint_tokens,
                        cancel_latency_seconds=(time.perf_counter()-job.cancel_requested if job.cancel_requested is not None else None),
                        cache_retained_bytes=self.prefix_cache.bytes,
                        gpu_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else None,
                        input_tokens=job.input_tokens,output_tokens=emitted_count,queue_seconds=queue_seconds,
                        output_budget=options['max_tokens'],output_budget_policy=self.output_budget_policy,
                        tokenization_seconds=tokenization_seconds,first_output_seconds=first_output_seconds,
                        server_seconds=time.perf_counter()-job.received,stream=options['stream'],
                        error={k:v for k,v in public_error(job.start_error).items() if k in ('code','tool','param')}
                              if job.start_error else None))
                except OSError:LOG.exception('Unable to record request %s audit',job.id)
            job.started.set()
            self._release(job)


class ContinuousRequest:
    """API framing and accounting for one private resumable model session."""
    def __init__(self,runtime,job):
        self.runtime=runtime;self.job=job;self.options=job.options
        self.begin=time.perf_counter();self.queue_seconds=self.begin-job.received
        self.tokenization_seconds=0.;self.first_output_seconds=None
        self.emitted_count=0;self.generation=None;self.framer=None;self.reasoning=None;self.closed=False
        self.output_decoder=None

    @property
    def phase(self):return self.generation.phase if self.generation else 'starting'

    @property
    def done(self):return bool(self.generation and self.generation.done)

    def start(self):
        runtime,job=self.runtime,self.job
        with runtime.lock:
            if time.perf_counter()>=job.queue_deadline and not job.cancel.is_set():
                job.cancel_requested=time.perf_counter();job.cancel_reason='queue_timeout'
                job.cancel.set();job.start_error=QueueDeadlineExceeded('The queue wait deadline was exceeded. Retry shortly.')
            job.executing.set()
        if job.cancel.is_set():raise GenerationCancelled('queued')
        job.phase='tokenization'
        if job.preparation is not None:
            prepared=job.preparation.result()
            ids=prepared.ids
            self.tokenization_seconds=prepared.tokenization_seconds
            self.options['max_tokens']=prepared.output_budget
            job.preparation=None
        else:
            begin=time.perf_counter()
            ids=request_token_ids(runtime.tokenizer,self.options)
            self.tokenization_seconds=time.perf_counter()-begin
        job.input_tokens=len(ids)
        if job.cancel.is_set():raise GenerationCancelled('tokenization')
        self.options['max_tokens']=runtime.resolve_output_budget(ids,self.options['max_tokens'])
        if not runtime.context_pool.try_reserve(job.id,len(ids),self.options['max_tokens']):
            raise RuntimeError('Context reservation was not acquired before model admission')
        def delta(value):
            if self.first_output_seconds is None and any(value.get(k) for k in ('content','reasoning_content','tool_calls')):
                self.first_output_seconds=time.perf_counter()-job.received
            if self.options['stream']:job.events.put(('delta',value))
        starts,self.reasoning=request_reasoning(runtime.tokenizer,ids,self.options,runtime.mode,runtime.eos)
        self.framer=OutputFramer(self.options['tools'],delta,starts_in_reasoning=starts)
        self.output_decoder=IncrementalOutputDecoder(runtime.tokenizer,runtime.eos)
        def progress(phase,completed,checkpoint):
            job.phase=phase;job.completed_prefill_tokens=completed;job.checkpoint_tokens=checkpoint
        extra=dict(should_cancel=job.cancel.is_set) if runtime.mode=='continuous_mtp' else {}
        self.generation=runtime.batch_runner.start(ids,self.options['max_tokens'],runtime.eos,PREFILL_CHUNK,
                                                 on_progress=progress,**extra)
        job.started.set()

    def prefill(self):self.runtime.batch_runner.prefill(self.generation)

    def decode(self):
        forced,limit=self.reasoning.plan()
        if forced is not None:
            self.runtime.batch_runner.force_next(self.generation,forced)
        elif limit is not None:
            self.runtime.batch_runner.decode([self.generation],emission_limit=limit)
        else:
            self.runtime.batch_runner.decode([self.generation])

    def after_step(self):
        # Called separately per request after all shared batch lanes commit.
        self.generation.progress()
        if self.job.cancel.is_set():
            raise GenerationCancelled(self.job.phase,self.job.completed_prefill_tokens,self.job.checkpoint_tokens)
        tokens=self.generation.generated
        if len(tokens)==self.emitted_count:return
        new_tokens=tokens[self.emitted_count:]
        self.emitted_count=len(tokens)
        raw=self.output_decoder.feed(new_tokens)
        self.framer.feed(raw)
        self.reasoning.observe(tokens)
        if self.runtime.mode=='continuous_mtp' and not self.generation.done:
            reason=('protocol_end_marker' if self.framer.mode=='end' else
                    'reasoning_reopened' if getattr(self.reasoning,'reasoning_reopened',False) else None)
            if reason is not None and self.runtime.batch_runner.stop(self.generation,reason):
                self.generation.progress()

    def finish(self,error=None):
        # Even a failure while constructing the audit (before normal response
        # handling) must clean up private state and return the reservation.
        runtime,job=self.runtime,self.job
        try:
            return self._finish_response(error)
        except BaseException:
            if not job.released:
                job.start_error=RuntimeError('The model request could not be finalized')
                job.events.put(('error','The model request could not be finalized.'))
            raise
        finally:
            if not job.released:
                try:
                    if self.generation and not self.closed:
                        runtime.batch_runner.close(self.generation)
                        self.closed=True
                except BaseException:
                    runtime.fatal=True;runtime.ready=False
                    raise
                finally:
                    self.generation=None;self.framer=None;self.reasoning=None;self.output_decoder=None
                    self.options={};job.options={};job.started.set();runtime._release(job)

    def _finish_response(self,error=None):
        runtime,job=self.runtime,self.job
        if error is None and job.cancel.is_set():
            error=GenerationCancelled(job.phase,job.completed_prefill_tokens,job.checkpoint_tokens)
        # Cancellation can happen after the runner commits a token and before
        # after_step() frames it. Account for that committed step in the audit.
        if self.reasoning is not None and self.generation is not None:
            self.reasoning.observe(self.generation.generated)
        status='completed';response=None
        audit=dict(id=job.id,created=job.created,input_tokens=job.input_tokens,
            output_budget=self.options['max_tokens'],output_budget_policy=runtime.output_budget_policy,
            output_tokens=len(self.generation.generated) if self.generation else 0,
            stream=self.options['stream'],execution_mode=runtime.mode,
            queue_seconds=self.queue_seconds,tokenization_seconds=self.tokenization_seconds,
            first_output_seconds=self.first_output_seconds)
        if self.reasoning is not None:
            audit['reasoning']=self.reasoning.snapshot()
        try:
            if error is not None:
                if isinstance(error,GenerationCancelled):
                    # A paired MTP chunk can publish its checkpoint and then
                    # cancel before after_step() reports progress to the job.
                    if error.phase!='request':job.phase=error.phase
                    job.completed_prefill_tokens=max(job.completed_prefill_tokens,error.completed_prefill_tokens)
                    job.checkpoint_tokens=max(job.checkpoint_tokens,error.checkpoint_tokens)
                    status='queue_timeout' if job.cancel_reason=='queue_timeout' else 'cancelled'
                    job.events.put(('cancelled',{}))
                    return None
                status='error';job.start_error=error
                if isinstance(error,RuntimeError) and any(term in str(error).lower() for term in ('cuda','nonfinite')):
                    runtime.fatal=True;runtime.ready=False
                audit['error']=record_failure(job,error)
                raise error
            row=self.generation.result(runtime.prefix_cache.bytes)
            # Drain and release model state before publishing terminal success.
            runtime.batch_runner.close(self.generation)
            self.closed=True
            self.framer.feed(self.output_decoder.finish(row['generated_ids']),final=True)
            finish='tool_calls' if self.framer.calls else row['finish_reason']
            usage=dict(prompt_tokens=job.input_tokens,completion_tokens=row['output_tokens'],
                total_tokens=job.input_tokens+row['output_tokens'],
                prompt_tokens_details=dict(cached_tokens=row['prefix_cache']['reused_tokens']))
            if self.reasoning.enabled:
                usage['completion_tokens_details']=dict(reasoning_tokens=self.reasoning.used)
            response=dict(id=job.id,object='chat.completion',created=job.created,model=MODEL,
                choices=[dict(index=0,message=self.framer.message(),finish_reason=finish)],usage=usage)
            audit.update({k:v for k,v in row.items() if k!='generated_ids'})
            audit['finish_reason']=finish
            job.events.put(('done',dict(finish_reason=finish,usage=usage)))
            return response
        except Exception as failure:
            # Includes final framing errors, which must also terminate this SSE.
            if status=='completed':
                status='error'
                audit['error']=record_failure(job,failure)
            if isinstance(failure,RuntimeError) and any(term in str(failure).lower() for term in ('cuda','nonfinite')):
                runtime.fatal=True;runtime.ready=False
            raise
        finally:
            try:
                if self.generation and not self.closed:runtime.batch_runner.close(self.generation)
            except BaseException:
                # A failed cleanup is an unavailable engine, not reusable quota.
                runtime.fatal=True;runtime.ready=False
                raise
            finally:
                audit.update(status=status,phase=job.phase,completed_prefill_tokens=job.completed_prefill_tokens,
                    checkpoint_tokens=job.checkpoint_tokens,cache_retained_bytes=runtime.prefix_cache.bytes,
                    first_output_seconds=self.first_output_seconds,server_seconds=time.perf_counter()-job.received,
                    cancel_latency_seconds=time.perf_counter()-job.cancel_requested if job.cancel_requested is not None else None)
                try:runtime.append_audit(audit)
                except OSError:LOG.exception('Unable to record request %s audit',job.id)
                finally:
                    self.generation=None;self.framer=None;self.reasoning=None;self.options={};job.options={}
                    self.output_decoder=None
                    job.started.set();runtime._release(job)


def create_app(runtime,key):
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST','OPTIONS'],
                       allow_headers=['Authorization','Content-Type'])

    def authorize(request):
        header=request.headers.get('authorization','')
        token=header[7:] if header.lower().startswith('bearer ') else ''
        if not hmac.compare_digest(token.encode(),key.encode()):raise HTTPException(401,'Invalid API key.')

    @app.exception_handler(HTTPException)
    async def http_error(request,error):
        kind='authentication_error' if error.status_code==401 else 'invalid_request_error' if error.status_code<500 else 'server_error'
        value=error.detail if isinstance(error.detail,dict) else dict(
            message=str(error.detail),type=kind,param=None,code=error.status_code)
        return JSONResponse(dict(error=value),
                            status_code=error.status_code,headers=error.headers)

    @app.get('/health')
    async def health():
        return JSONResponse(dict(status='ok' if runtime.ready else 'unavailable',model=MODEL),status_code=200 if runtime.ready else 503)

    @app.get('/v1/models')
    async def models(request:Request):
        authorize(request)
        return dict(object='list',data=[dict(id=MODEL,object='model',created=1790812800,owned_by='spark',
            context_window=getattr(runtime,'max_context',MAX_CONTEXT),
            reasoning_efforts=REASONING_EFFORTS if getattr(runtime,'mode',None)=='continuous_mtp' else {},
            max_output_tokens=getattr(runtime,'max_output',MAX_OUTPUT) or None,
            output_budget_policy=getattr(runtime,'output_budget_policy',
                'remaining_context' if getattr(runtime,'max_output',MAX_OUTPUT)==0 else 'fixed_limit'))])

    @app.get('/v1/scheduler')
    async def scheduler_status(request:Request):
        authorize(request)
        scheduler=getattr(runtime,'scheduler',None)
        mode=getattr(runtime,'mode','serial_mtp')
        return dict(execution_mode=mode,
            context_window_per_request=getattr(runtime,'max_context',MAX_CONTEXT),
            max_output_tokens=getattr(runtime,'max_output',MAX_OUTPUT) or None,
            output_budget_policy=getattr(runtime,'output_budget_policy',
                'remaining_context' if getattr(runtime,'max_output',MAX_OUTPUT)==0 else 'fixed_limit'),
            aggregate_context_budget=getattr(runtime,'total_context',getattr(runtime,'max_active',1)*getattr(runtime,'max_context',MAX_CONTEXT)),
            context_pool=runtime.context_pool.snapshot() if hasattr(runtime,'context_pool') else None,
            prepare_workers=getattr(runtime,'prepare_workers',0),
            mtp_enabled=mode in ('serial_mtp','continuous_mtp'),
            reasoning_efforts=REASONING_EFFORTS if mode=='continuous_mtp' else {},
            scheduling='decode_priority_batch' if getattr(runtime,'mtp_batch',False) else 'cooperative_round_robin' if mode=='continuous_mtp' else mode,
            batched_verification=getattr(runtime,'mtp_batch',False),experimental=mode=='continuous_mtp',
            **(scheduler.snapshot() if scheduler else dict(max_active=1,
                max_queued=getattr(runtime,'max_queued',1),events=[])))

    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        authorize(request)
        try:
            body=await request.json()
            options=normalize_request(body,MODEL,getattr(runtime,'max_output',MAX_OUTPUT))
            if options.get('thinking_budget') is not None and getattr(runtime,'mode',None)!='continuous_mtp':
                raise RequestError('Finite thinking budgets currently require continuous_mtp.')
        except (ValueError,TypeError,AttributeError) as error:
            if isinstance(error,(RequestError,ToolSchemaError)):
                raise request_http_error(error) from error
            raise HTTPException(400,'Invalid request JSON or options.') from error
        job=runtime.submit(options)
        def request_failure():
            if isinstance(job.start_error,QueueDeadlineExceeded):
                raise HTTPException(503,str(job.start_error),headers={'Retry-After':'1'})
            if job.start_error:
                raise request_http_error(job.start_error)
            if job.cancel.is_set():raise HTTPException(499,'Client disconnected.')
        if not options['stream']:
            try:
                while not job.future.done():
                    if await request.is_disconnected():
                        runtime.cancel_job(job)
                        raise HTTPException(499,'Client disconnected.')
                    runtime.expire_queued(job)
                    await asyncio.sleep(.05)
                request_failure()
                try:return job.future.result()
                except (RequestError,ToolSchemaError,ToolCallError) as error:raise request_http_error(error) from error
                except FutureCancelledError as error:raise HTTPException(499,'Client disconnected.') from error
                except Exception as error:raise HTTPException(502,'The model request failed.') from error
            finally:runtime.cancel_job(job)
        try:
            while not job.started.is_set():
                if await request.is_disconnected():
                    runtime.cancel_job(job)
                    raise HTTPException(499,'Client disconnected.')
                runtime.expire_queued(job)
                await asyncio.sleep(.05)
            request_failure()
        except BaseException:
            runtime.cancel_job(job)
            raise

        def chunk(delta,finish=None):
            return dict(id=job.id,object='chat.completion.chunk',created=job.created,model=MODEL,
                        choices=[dict(index=0,delta=delta,finish_reason=finish)])
        def sse(value):return 'data: '+json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n\n'

        async def stream():
            try:
                yield sse(chunk(dict(role='assistant',content='')))
                while True:
                    if await request.is_disconnected():runtime.cancel_job(job); break
                    try:kind,value=await asyncio.to_thread(job.events.get,True,1)
                    except queue.Empty:
                        yield ': keep-alive\n\n'; continue
                    if kind=='delta':yield sse(chunk(value))
                    elif kind=='done':
                        yield sse(chunk({},value['finish_reason']))
                        # This dedicated engine always returns cache accounting.
                        # Clients must not need an opt-in to see reused tokens.
                        yield sse(dict(id=job.id,object='chat.completion.chunk',created=job.created,model=MODEL,
                                       choices=[],usage=value['usage']))
                        yield 'data: [DONE]\n\n'; break
                    elif kind=='error':
                        error=value if isinstance(value,dict) else dict(message=value,type='server_error',code=502,param=None)
                        yield sse(dict(id=job.id,error=error))
                        yield 'data: [DONE]\n\n'; break
                    elif kind=='cancelled':break
            finally:runtime.cancel_job(job)
        return StreamingResponse(stream(),media_type='text/event-stream',
            headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no','X-Request-ID':job.id})
    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    key=Path(os.environ['SPARK_KEY_FILE']).read_text().strip()
    if len(key)<32:raise ValueError('Expected a strong nonempty API key.')
    runtime=Runtime()
    try:
        runtime.executor.submit(runtime.load).result()
        uvicorn.run(create_app(runtime,key),host='0.0.0.0',port=8000,access_log=False)
    finally:runtime.close()


if __name__=='__main__':main()
