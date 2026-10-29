"""Ember entrypoint: install the SM121 optimizers, then start the API server.

The optimizers write a copy of the MoE kernel module and generated CUDA-graph
helpers into a fresh writable workspace (SPARK_LAB_DIR, default /lab). Keep the
source tree read-only and give every start its own empty workspace.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
LAB=Path(os.environ.get('SPARK_LAB_DIR','/lab'))
_installed=False

def prepare_leaf(root,work):
    if not work.is_dir():raise RuntimeError('Expected a writable workspace directory: '+str(work))
    source=root/'moe_upfirst.py'
    target=work/'moe_upfirst.py'
    if target.exists():
        if target.is_symlink() or target.read_bytes()!=source.read_bytes():
            raise RuntimeError('Existing MoE kernel copy differs; use a fresh workspace')
    else:
        with target.open('xb') as stream:stream.write(source.read_bytes())

def install():
    global _installed
    if _installed:raise RuntimeError('Refusing to install runtime patches twice')
    prepare_leaf(ROOT,LAB)
    receipt=LAB/'install.json'
    if receipt.exists():raise RuntimeError('Refusing to reuse an optimizer installation workspace')
    # Optional: point at an unpacked nvidia-cutlass-dsl tree instead of the installed package.
    dsl=os.environ.get('SPARK_CUTLASS_DSL_DIR')
    if dsl:sys.path.insert(0,dsl)
    sys.path.insert(0,str(ROOT))
    adapter=importlib.import_module('moe_adapter')
    state=importlib.import_module('state_opt')
    # Order matters: the state optimizer patches graphs built on the MoE adapter.
    adapter.install();adapter.MODE='candidate';adapter.SHADOW=False
    state.install();state.MODE='abc'
    result=dict(optimizer='abc',moe_backend='candidate',model_loaded=False)
    with receipt.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    _installed=True
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-install',action='store_true',help='Install patches and exit without loading weights or serving')
    parser.add_argument('--speculation',choices=('mtp','tree'),help='Startup decoding strategy; tree is experimental, default mtp')
    args=parser.parse_args()
    if args.speculation is not None:os.environ['SPARK_SPECULATION']=args.speculation
    result=install()
    print('OPTIMIZERS_INSTALLED',json.dumps(result),flush=True)
    if args.check_install:return
    from api_server import main as serve
    serve()

if __name__=='__main__':main()
