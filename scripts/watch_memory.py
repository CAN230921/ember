#!/usr/bin/env python3
"""Stop only the newly created, labelled Ember container on host memory pressure."""
import argparse,json,re,signal,subprocess,time
from pathlib import Path

def pressured(available_kib,swap_kib,baseline_kib):
    return available_kib<20*1024**2 or swap_kib>1024**2 or swap_kib>baseline_kib+128*1024

def owned(info,cid):
    return info['Id']==cid and (info['Config'].get('Labels') or {}).get('io.ember.engine')=='true'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('container');p.add_argument('baseline_swap_kib',type=int);p.add_argument('ready',type=Path);a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',a.container):p.error('Expected the full newly created container ID')
    def interrupt(signum,frame):raise KeyboardInterrupt()
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupt)
    verified=False;stopped=False
    try:
        while True:
            result=subprocess.run(['docker','inspect',a.container],check=True,capture_output=True,text=True,timeout=10)
            info=json.loads(result.stdout)[0]
            if not owned(info,a.container):raise RuntimeError('Container identity or ownership label mismatch')
            verified=True
            if not info['State']['Running']:stopped=True;return
            mem={k:int(v.split()[0]) for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
            swap=mem['SwapTotal']-mem['SwapFree'];available=mem['MemAvailable']
            print(json.dumps({'available_kib':available,'swap_kib':swap}),flush=True)
            if pressured(available,swap,a.baseline_swap_kib):raise RuntimeError('Host memory/swap guard tripped')
            if not a.ready.exists():a.ready.write_text(a.container+'\n')
            time.sleep(2)
    finally:
        if verified and not stopped:
            subprocess.run(['docker','stop','--timeout','30',a.container],check=True,timeout=45)

if __name__=='__main__':main()
