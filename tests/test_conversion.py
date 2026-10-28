import json
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from convert_checkpoint import Reader, convert
from engine.ops.m0_refs import gptq_int8_dequant, fp8_blockwise_dequant
from engine.ops.ple_lookup import dequant_rows


def source_fixture(root):
    root.mkdir();torch.manual_seed(23)
    prefix='model.language_model.layers.0.'
    data={prefix+'linear_attn.in_proj_qkv.weight':torch.randn(128,128).bfloat16(),
          'lm_head.weight':torch.randn(16,128).bfloat16(),
          prefix+'mlp.experts.0.gate_proj.weight':torch.randint(0,256,(128,64),dtype=torch.uint8),
          prefix+'ple.ple_embedding.ngram_embedding.shard_0.weight':(torch.randn(21,160)*3).to(torch.float8_e4m3fn),
          prefix+'ple.ple_embedding.ngram_embedding.weight_scale':torch.tensor([.125],dtype=torch.bfloat16)}
    for expert in range(2):
        for proj in ('gate_proj','up_proj','down_proj'):
            p=f'mtp.layers.0.mlp.experts.{expert}.{proj}'
            data[p+'.weight']=(torch.randn(128,128)*(expert+1)).to(torch.float8_e4m3fn)
            data[p+'.weight_scale_inv']=torch.tensor([[.25]])
    save_file(data,str(root/'source.safetensors'))
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{n:'source.safetensors' for n in data}}))
    (root/'config.json').write_text(json.dumps({'text_config':{'num_experts':2}}))
    return data


def test_streamed_conversion_preserves_layout_and_source(tmp_path):
    source=tmp_path/'source';data=source_fixture(source);before={p.name:p.read_bytes() for p in source.iterdir()}
    output=tmp_path/'out';report=convert(source,output,chunk_rows=7,strict=False);r=Reader(output)
    assert report['status']=='complete'
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    prefix='model.language_model.layers.0.'
    assert torch.equal(r.tensor(prefix+'mlp.experts.0.gate_proj.weight'),data[prefix+'mlp.experts.0.gate_proj.weight'])
    q=r.tensor(prefix+'linear_attn.in_proj_qkv.weight');s=r.tensor(prefix+'linear_attn.in_proj_qkv.weight_scale_inv')
    ref=fp8_blockwise_dequant(q,s);w=data[prefix+'linear_attn.in_proj_qkv.weight'].float()
    assert (ref-w).norm()/w.norm()<.05
    head=gptq_int8_dequant(r.tensor('lm_head.qweight'),r.tensor('lm_head.qzeros'),r.tensor('lm_head.scales')).T
    assert (head-data['lm_head.weight'].float()).norm()/data['lm_head.weight'].float().norm()<.02
    for expert in range(2):
        bank=r.tensor('mtp.layers.0.mlp.experts.gate_up_proj')[expert]
        for i,proj in enumerate(('gate_proj','up_proj')):
            raw=data[f'mtp.layers.0.mlp.experts.{expert}.{proj}.weight'].float()*.25
            assert torch.equal(bank[i*128:(i+1)*128],raw.bfloat16())
    raw=r.tensor(prefix+'ple.ple_embedding.ngram_embedding.shard_0.weight')
    assert raw.shape==(21,42)
    decoded=dequant_rows(raw,'tern_g16_none',.125)
    assert torch.isfinite(decoded).all()
    with pytest.raises(FileExistsError):convert(source,output,strict=False)


def test_rejects_unsupported_schema_without_creating_output(tmp_path):
    source=tmp_path/'source';source_fixture(source);output=tmp_path/'out'
    with pytest.raises(ValueError,match='only supports'):convert(source,output)
    assert not output.exists()


@pytest.mark.parametrize('filename',['../source.safetensors','/source.safetensors'])
def test_source_index_cannot_escape_or_overwrite_source(tmp_path,filename):
    source=tmp_path/'source';source_fixture(source)
    index=source/'model.safetensors.index.json';data=json.loads(index.read_text())
    data['weight_map'][next(iter(data['weight_map']))]=filename
    index.write_text(json.dumps(data));before={p.name:p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError,match='root-local'):convert(source,tmp_path/'out',strict=False)
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    assert not (tmp_path/'out').exists()
