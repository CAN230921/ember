#!/usr/bin/env python3
"""Convert the official NVFP4 checkpoint directly, without a private layout template.

Original NVFP4 experts are copied. Main BF16 projections become block128 FP8,
the BF16 output head becomes symmetric GPTQ8/group128, MTP FP8 experts become
BF16 banks, and FP8 PLE rows become tern_g16_none. Never overwrites a directory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from engine.ops.ple_lookup import quantize_tern

PROJECTION=re.compile(r'model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))\.weight$')
MTP=re.compile(r'mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$')
PLE=re.compile(r'.*\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$')
DT={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'F8_E4M3':torch.float8_e4m3fn,'U8':torch.uint8,'I32':torch.int32}
SIZE={'BF16':2,'F32':4,'F16':2,'F8_E4M3':1,'U8':1,'I32':4,'I64':8}
BLOCK=8*1024**2


class Reader:
    def __init__(self,root):
        self.root=Path(root);self.index=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map'];self.headers={}
        if any(not isinstance(name,str) or Path(name).name!=name or not name.endswith('.safetensors')
               for name in self.index.values()):
            raise ValueError('Expected root-local safetensors filenames in the source index')
    def meta(self,name):
        filename=self.index[name]
        if filename not in self.headers:
            with (self.root/filename).open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];self.headers[filename]=(json.loads(f.read(n)),8+n)
        header,base=self.headers[filename];return self.root/filename,header[name],base
    def raw(self,name,start=0,length=None):
        path,m,base=self.meta(name);total=m['data_offsets'][1]-m['data_offsets'][0]
        length=total-start if length is None else length
        if start<0 or length<0 or start+length>total:raise ValueError('Tensor slice outside source')
        with path.open('rb') as f:
            f.seek(base+m['data_offsets'][0]+start);b=f.read(length)
        if len(b)!=length:raise EOFError(name)
        return b
    def tensor(self,name):
        _,m,_=self.meta(name)
        return torch.frombuffer(bytearray(self.raw(name)),dtype=DT[m['dtype']]).reshape(m['shape'])


def fp8_block128(weight):
    n,k=weight.shape
    if n%128 or k%128:raise ValueError('FP8 projection dimensions must be divisible by 128')
    w=weight.float().reshape(n//128,128,k//128,128)
    scale=w.abs().amax((1,3),keepdim=True).clamp_min(1e-12)/448
    q=(w/scale).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.reshape(n,k).contiguous(),scale[:,0,:,0].contiguous()


def int8_head(weight):
    n,k=weight.shape
    if k%128 or n%4:raise ValueError('Unsupported output head dimensions')
    qw=np.empty((k//4,n),dtype=np.int32);sc=np.empty((k//128,n),dtype=np.float16)
    for start in range(0,n,4096):
        end=min(n,start+4096);g=weight[start:end].float().T.reshape(k//128,128,end-start)
        mn,mx=g.amin(1),g.amax(1);picked=torch.where(-mn>mx,mn,mx)
        scale=(picked/-128).to(torch.float16);scale=torch.where(scale==0,torch.ones_like(scale),scale)
        q=(torch.round(g/scale.float()[:,None,:])+128).clamp(0,255).to(torch.uint8)
        packed=q.reshape(k//4,4,end-start).permute(0,2,1).contiguous().numpy().view(np.int32).reshape(k//4,end-start)
        qw[:,start:end]=packed;sc[:,start:end]=scale.numpy()
    return {'lm_head.qweight':torch.from_numpy(qw),'lm_head.qzeros':torch.full((k//128,n//4),0x7f7f7f7f,dtype=torch.int32),'lm_head.scales':torch.from_numpy(sc)}


def tensor_bytes(tensor):return memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B')


def dequant_mtp(reader,name):
    q=reader.tensor(name);scale=reader.tensor(name.removesuffix('.weight')+'.weight_scale_inv')
    n,k=q.shape
    if q.dtype!=torch.float8_e4m3fn or tuple(scale.shape)!=(n//128,k//128):raise ValueError('Unsupported MTP FP8 schema: '+name)
    return (q.float().reshape(n//128,128,k//128,128)*scale.float()[:,None,:,None]).reshape(n,k).to(torch.bfloat16)


def write_shard(output,filename,specs,index):
    """Each producer yields bounded byte chunks; even fused expert banks stream."""
    if Path(filename).name!=filename:raise ValueError('Output shard must be inside the output directory')
    header={'__metadata__':{'format':'pt','ple_format':'tern_g16_none'}};offset=0
    for name,dtype,shape,producer in specs:
        n=math.prod(shape)*SIZE[dtype];header[name]={'dtype':dtype,'shape':list(shape),'data_offsets':[offset,offset+n]};offset+=n
    blob=json.dumps(header,separators=(',',':')).encode();blob+=b' '*(-len(blob)%8)
    temp=output/(filename+'.partial');digest=hashlib.sha256()
    with temp.open('xb') as f:
        f.write(struct.pack('<Q',len(blob)));f.write(blob)
        for name,dtype,shape,producer in specs:
            size=0
            for part in producer():f.write(part);digest.update(part);size+=len(part)
            if size!=math.prod(shape)*SIZE[dtype]:raise ValueError('Wrong encoded size: '+name)
            index[name]=filename
    temp.rename(output/filename)
    return {'data_sha256':digest.hexdigest(),'tensor_bytes':offset}


def convert(source,output,ple_device='cpu',chunk_rows=65536,strict=True):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite output directory')
    reader=Reader(source);names=set(reader.index);config=json.loads((source/'config.json').read_text());tc=config['text_config']
    projections={n for n in names if PROJECTION.fullmatch(n)}
    if not projections or 'lm_head.weight' not in names:raise ValueError('Expected the original official BF16-projection/NVFP4 checkpoint')
    for name in projections|{'lm_head.weight'}:
        if reader.meta(name)[1]['dtype']!='BF16':raise ValueError('Source already converted or unsupported: '+name)
    ple=sorted(n for n in names if PLE.fullmatch(n))
    if not ple:raise ValueError('PLE table missing')
    if strict and (len(projections)!=300 or len(ple)!=128 or tc.get('hidden_size')!=2560 or
                   tc.get('num_hidden_layers')!=48 or tc.get('num_experts')!=512):
        raise ValueError('This converter only supports the documented Qwen3.8 Flash Next schema')
    scale_name=ple[0].rsplit('.shard_',1)[0]+'.weight_scale'
    embedding_scale=reader.tensor(scale_name).to(torch.bfloat16);gs=float(embedding_scale.float().item())
    mtp_names={n for n in names if MTP.fullmatch(n)}
    experts=sorted({int(MTP.fullmatch(n).group(1)) for n in mtp_names})
    if experts!=list(range(tc['num_experts'])):raise ValueError('Incomplete MTP experts')
    output.mkdir(parents=True);new_index={};report={'status':'converting','source_index_sha256':hashlib.sha256((source/'model.safetensors.index.json').read_bytes()).hexdigest(),'ple_format':'tern_g16_none','projections':len(projections),'mtp_experts':len(experts),'shards':{}}
    groups={}
    for name,filename in reader.index.items():groups.setdefault(filename,[]).append(name)
    head_cache=None
    for filename,group in sorted(groups.items()):
        specs=[];cache={}
        for name in sorted(group):
            _,meta,_=reader.meta(name);dtype=meta['dtype'];shape=meta['shape']
            if name in mtp_names:continue
            if name in projections:
                def quantized(n=name):
                    q,s=fp8_block128(reader.tensor(n));cache[n]=s;yield tensor_bytes(q)
                specs.append((name,'F8_E4M3',shape,quantized))
                def scaled(n=name):yield tensor_bytes(cache.pop(n))
                specs.append((name.removesuffix('.weight')+'.weight_scale_inv','F32',[shape[0]//128,shape[1]//128],scaled))
            elif name=='lm_head.weight':
                head_cache=int8_head(reader.tensor(name))
                for key,t in head_cache.items():
                    dt={'torch.int32':'I32','torch.float16':'F16'}[str(t.dtype)]
                    specs.append((key,dt,list(t.shape),lambda t=t:iter([tensor_bytes(t)])))
            elif PLE.fullmatch(name):
                if dtype!='F8_E4M3' or shape[1]!=160:raise ValueError('Unsupported original PLE schema')
                def tern(n=name,rows=shape[0]):
                    for start in range(0,rows,chunk_rows):
                        count=min(chunk_rows,rows-start);raw=reader.raw(n,start*160,count*160)
                        x=torch.frombuffer(bytearray(raw),dtype=torch.float8_e4m3fn).reshape(count,160).to(ple_device).to(torch.bfloat16)
                        x=(x*embedding_scale.to(ple_device)).float()
                        encoded=quantize_tern(x,gs,16,False).cpu();yield tensor_bytes(encoded)
                specs.append((name,'U8',[shape[0],42],tern))
            else:
                def copied(n=name,size=meta['data_offsets'][1]-meta['data_offsets'][0]):
                    for pos in range(0,size,BLOCK):yield reader.raw(n,pos,min(BLOCK,size-pos))
                specs.append((name,dtype,shape,copied))
        if specs:
            report['shards'][filename]=write_shard(output,filename,specs,new_index)
            print(json.dumps({'completed_shard':filename,'tensors':len(new_index)}),flush=True)
        head_cache=None
    for bank,projs in (('gate_up_proj',('gate_proj','up_proj')),('down_proj',('down_proj',))):
        prefix='mtp.layers.0.mlp.experts.'
        shape=reader.meta(prefix+'0.'+projs[0]+'.weight')[1]['shape'];shape=[len(experts),shape[0]*len(projs),shape[1]]
        def bank_data(projs=projs):
            for expert in experts:
                for proj in projs:yield tensor_bytes(dequant_mtp(reader,prefix+str(expert)+'.'+proj+'.weight'))
        filename='ember-mtp-'+bank+'.safetensors';name=prefix+bank
        report['shards'][filename]=write_shard(output,filename,[(name,'BF16',shape,bank_data)],new_index)
    tc['num_experts_per_tok']=10;tc['ple_embedding_dtype']='float8_e4m3fn'
    for filename in ('tokenizer.json','tokenizer_config.json','generation_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt'):
        if (source/filename).is_file():shutil.copyfile(source/filename,output/filename)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(x['tensor_bytes'] for x in report['shards'].values())},'weight_map':new_index},indent=2)+'\n')
    report.update(status='complete',tensors=len(new_index));(output/'ember-conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ple-device',choices=('cpu','cuda'),default='cpu');p.add_argument('--chunk-rows',type=int,default=65536);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    if a.chunk_rows<=0:p.error('--chunk-rows must be positive')
    torch.set_num_threads(a.threads);convert(a.source,a.output,a.ple_device,a.chunk_rows)
