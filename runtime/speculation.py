"""Startup-only speculation selection; the existing MTP path is the default."""
import os


def settings(mode=None):
    mode=os.environ.get('SPARK_SPECULATION','mtp') if mode is None else mode
    if mode not in ('mtp','tree'):raise ValueError('SPARK_SPECULATION must be mtp or tree')
    if mode=='mtp':return dict(mode='mtp')
    if not os.environ.get('SPARK_DFLASH_DIR'):raise ValueError('Tree speculation requires SPARK_DFLASH_DIR')
    policy=os.environ.get('SPARK_TREE_POLICY','best_first')
    if policy not in ('beam','best_first'):raise ValueError('Unknown tree proposal policy')
    kernel=os.environ.get('SPARK_TREE_GDN_KERNEL','1')
    if kernel not in ('0','1'):raise ValueError('SPARK_TREE_GDN_KERNEL must be 0 or 1')
    return dict(mode='tree',policy=policy,max_nodes=8,max_leaves=4,gdn_kernel=kernel=='1',
                scheduling='cooperative_round_robin',batched_verification=False)
