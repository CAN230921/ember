from pathlib import Path
from unittest.mock import patch
from engine.ops.native_kernels import library_candidates


def test_qsa_library_honors_configured_directory():
    with patch.dict('os.environ', {'ENGINE_NATIVE_KERNEL_DIR': '/custom-kernels'}):
        assert library_candidates('qsa_topk')[0] == Path('/custom-kernels/_C_det.so')
        assert library_candidates('gemm')[0] == Path('/custom-kernels/_C_stable_libtorch.abi3.so')


def test_qsa_includes_repository_build_output():
    with patch.dict('os.environ', {}, clear=True):
        assert any(p.parent.name == 'native' and p.name == '_C_det.so'
                   for p in library_candidates('qsa_topk'))
