import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

s=importlib.util.spec_from_file_location('memory_guard',Path(__file__).resolve().parents[1]/'scripts/watch_memory.py')
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def test_host_guard_threshold_and_swap_growth():
    assert not g.pressured(21*1024**2,512*1024,512*1024)
    assert g.pressured(19*1024**2,512*1024,512*1024)
    assert g.pressured(21*1024**2,641*1024,512*1024)

def test_unrelated_or_replaced_container_is_not_owned():
    cid='a'*64
    assert g.owned({'Id':cid,'Config':{'Labels':{'io.ember.engine':'true'}}},cid)
    assert not g.owned({'Id':'b'*64,'Config':{'Labels':{'io.ember.engine':'true'}}},cid)
    assert not g.owned({'Id':cid,'Config':{'Labels':{}}},cid)


@pytest.mark.parametrize('labelled',[True,False])
def test_monitor_stops_only_its_labelled_container_on_pressure(monkeypatch,tmp_path,labelled):
    cid='a'*64;calls=[]
    info={'Id':cid,'Config':{'Labels':{'io.ember.engine':'true'} if labelled else {}},'State':{'Running':True}}
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps([info]),returncode=0)
    monkeypatch.setattr(g.subprocess,'run',run)
    monkeypatch.setattr('sys.argv',['guard',cid,'0',str(tmp_path/'ready')])
    monkeypatch.setattr(g.signal,'signal',lambda *a:None)
    monkeypatch.setattr(g.Path,'read_text',lambda p:'MemAvailable: 1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
    with pytest.raises(RuntimeError):g.main()
    stops=[cmd for cmd in calls if cmd[:2]==['docker','stop']]
    assert stops==[['docker','stop','--timeout','30',cid]] if labelled else stops==[]
