#!/usr/bin/env python3
"""Check the complete runtime contract without reading credentials or model weights."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true',help='Also verify an available SM121 CUDA device')
    args=parser.parse_args();errors=[];versions={}
    if os.environ.get('SPARK_CUTLASS_DSL_DIR'):sys.path.insert(0,os.environ['SPARK_CUTLASS_DSL_DIR'])
    for module,package in {'torch':'torch','triton':'triton','flashinfer':'flashinfer-python',
            'transformers':'transformers','fastapi':'fastapi','uvicorn':'uvicorn',
            'jsonschema':'jsonschema','numpy':'numpy','safetensors':'safetensors','cutlass':'nvidia-cutlass-dsl'}.items():
        try:
            loaded=importlib.import_module(module)
            versions[package]=importlib.metadata.version(package)
            if package=='nvidia-cutlass-dsl' and os.environ.get('SPARK_CUTLASS_DSL_DIR'):
                root=Path(os.environ['SPARK_CUTLASS_DSL_DIR']).resolve().parents[1]
                for dist in importlib.metadata.distributions(path=[str(root)]):
                    if dist.metadata['Name'].replace('_','-')=='nvidia-cutlass-dsl':versions[package]=dist.version
        except Exception as exc:errors.append(f'{package}: {type(exc).__name__}: {exc}')
    if versions.get('nvidia-cutlass-dsl') not in (None,'4.7.1'):
        errors.append('The measured MoE backend requires nvidia-cutlass-dsl 4.7.1')
    if 'torch' in versions:
        import torch
        if args.gpu and (not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,1)):
            errors.append('An available SM121 CUDA device is required')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
        try:
            from engine.ops.native_kernels import load_operator,library_candidates
            for kind in ('gemm','router','qsa_topk'):
                if args.gpu:load_operator(kind)
                elif not any(p.is_file() for p in library_candidates(kind)):
                    errors.append(f'Missing native library: {kind}')
        except Exception as exc:errors.append(f'native operators: {exc}')
    root=Path(__file__).resolve().parents[1]/'runtime'/'engine'/'ops'
    for name in ('fp8_quant','int8_head','int8_head_rows_v2','int8_head_three','shared_epilogue'):
        if not (root/f'lib{name}_sm121.so').is_file():errors.append(f'Missing built kernel: {name}')
    print(json.dumps({'versions':versions,'errors':errors,'passed':not errors},indent=2))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
