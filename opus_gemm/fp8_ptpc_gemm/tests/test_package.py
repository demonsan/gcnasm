# SPDX-License-Identifier: MIT
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import policy
import candidate

class PolicyTests(unittest.TestCase):
    def test_priority17(self):
        self.assertEqual(len(policy.SHAPES), 17)
        choices = [policy.select(*s) for s in policy.SHAPES]
        self.assertEqual([c.cfg for c in choices].count('c1'), 4)
        self.assertEqual([c.cfg for c in choices].count('c11'), 3)
        self.assertEqual([c.cfg for c in choices].count('c19'), 10)

        for (m, n, k), c in zip(policy.SHAPES, choices):
            if m == 16:
                self.assertEqual(c.requested_split, policy.M16_SPLITS[n, k])
            elif m == 32:
                self.assertEqual(c.requested_split, 3 if k == 6144 else 4)
            else:
                self.assertEqual(c.requested_split, 1)

    def test_partitions(self):
        for cfg in policy.CONFIGS:
            for k in (32, 160, 192, 256, 768, 6144, 8192):
                for req in (1, 2, 3, 4, 5, 7, 8, 64):
                    c = policy.select(32, 208, k, req, cfg)
                    ips, se = c.partition(k)
                    tiles = policy.ceil_div(k, c.bk)
                    self.assertGreater(ips, 0)
                    self.assertLessEqual(se, req)
                    self.assertGreater(tiles - (se - 1) * ips, 0)
                    self.assertGreaterEqual(se * ips, tiles)
                    if c.artifact == 'eightwave' and tiles >= 2:
                        self.assertEqual(ips % 2, 0)
        self.assertEqual(policy.select(16, 9216, 6144).workspace_shape(16, 9216, 6144), (8, 16, 9216))
        self.assertEqual(policy.select(32, 9856, 6144).workspace_shape(32, 9856, 6144), (3, 32, 9856))
        self.assertIsNone(policy.select(4096, 9216, 6144).workspace_shape(4096, 9216, 6144))
        self.assertEqual(policy.select(33, 208, 192, 2, 'c3').partition(192), (2, 1))

    def test_input_selector(self):
        c = policy.select(4096, 9216, 6144)
        def describe(**kw):
            return policy.describe(c, 4096, 9216, 6144, kw.get('a', 16), kw.get('w', 32), kw.get('y', 48), kw.get('lda', 6144), kw.get('ldc', 9216))
        self.assertEqual((describe()['tile'], describe()['path'], describe()['grid']), ((256, 192, 128), 'C19_256192', 768))
        self.assertEqual(describe(lda=6145)['path'], 'C3_U')
        self.assertEqual(describe(ldc=9224)['path'], 'C3_EPI_A')
        self.assertEqual(describe(w=33)['path'], 'C3_U')
        self.assertEqual(describe(y=50)['path'], 'C3_A')
        c = policy.select(8192, 9856, 6144)
        self.assertEqual(policy.describe(c, 8192, 9856, 6144, 16, 32, 48, 6144, 9856)['path'], 'C18_CONTRACT_ORDER')
        c = policy.select(17, 208, 160, 3, 'c11')
        self.assertEqual(policy.describe(c, 17, 208, 160, 16, 32, 48, 160, 208)['path'], 'C7_A')
        c = policy.select(32, 9216, 6144)
        self.assertEqual(policy.describe(c, 32, 9216, 6144, 16, 32, 48, 6144, 9216)['path'], 'C11_PC_A')

    def test_domain_and_disabled_experiments(self):
        for s in ((0, 16, 32), (1, 17, 32), (1, 16, 33), (1, 0, 32)):
            with self.assertRaises(ValueError): policy.select(*s)
        for req in (0, -1, True, 1.5, 2**31):
            with self.assertRaises(ValueError): policy.select(16, 64, 256, req)

        self.assertEqual(policy.select(16, 64, 256, 2**31 - 1).partition(256), (1, 1))
        with self.assertRaises(ValueError): policy.select(16, 64, 512, 2**31 - 1)
        for cfg in ('c4', 'c18f', 'c16', 'c17', 'unknown'):
            with self.assertRaises(ValueError): policy.select(16, 64, 256, cfg=cfg)

    def test_preshuffle_bijection_and_coordinates(self):
        for n, k in ((16, 32), (32, 160), (208, 192)):
            offsets = [policy.preshuffle_offset(i, j, k) for i in range(n) for j in range(k)]
            self.assertEqual(sorted(offsets), list(range(n * k)))

            source = [i * k + j for nb in range(n // 16) for kb in range(k // 32)
                      for h in range(2) for ni in range(16) for ki in range(16)
                      for i, j in [(nb * 16 + ni, kb * 32 + h * 16 + ki)]]
            for i in range(n):
                for j in range(k):
                    self.assertEqual(source[policy.preshuffle_offset(i, j, k)], i * k + j)

class PackageTests(unittest.TestCase):

    def test_local_includes_resolve(self):
        for p in list((ROOT / 'include/gfx950').glob('*.cuh')) + list(ROOT.glob('*_kernel.cc')):
            for include in re.findall(r'^\s*#include "([^"]+)"', p.read_text(encoding='utf-8'), re.M):
                if include != 'opus_gemm_utils.cuh':
                    self.assertTrue((ROOT / 'include/gfx950' / include).is_file(), (p.name, include))

    def test_python_syntax(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

    def test_plan_no_torch_import(self):
        code = 'import sys; import candidate, policy, gemm_host; assert "torch" not in sys.modules'

        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
        subprocess.run([sys.executable, 'gemm_host.py', '--plan-only', '-m', '32', '-n', '6144', '-k', '8192'], cwd=ROOT, check=True)

    def test_environment_restored_on_load_error(self):
        source = ROOT / 'bindings/splitk_candidate.py'
        fake = types.SimpleNamespace(_load=mock.Mock(side_effect=RuntimeError('fixture')))
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m: None))
        with mock.patch.object(candidate.Path, 'is_file', return_value=True), \
             mock.patch.object(candidate.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             mock.patch.object(candidate.importlib.util, 'module_from_spec', return_value=fake), \
             mock.patch.dict(candidate.os.environ, {'TASK_ROOT': 'previous-root', 'SO_BASENAME': 'previous.so'}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                candidate._load_snapshot('splitk', ROOT / 'build')
            self.assertEqual(candidate.os.environ['TASK_ROOT'], 'previous-root')
            self.assertEqual(candidate.os.environ['SO_BASENAME'], 'previous.so')

    def test_adapter_warmup_and_device_local_workspace(self):
        capture = [False]
        torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_current_stream_capturing=lambda: capture[0], device=lambda _: nullcontext()))
        made = []
        def maker(options, split_k):
            made.append((options['cfg'], split_k))
            def inner(a, w, sa, sb, out):
                inner.actual_cfg = options['cfg']
                return policy.select(*a.dimensions, split_k, options['cfg']).partition(a.dimensions[2])[1]
            inner.active_ws = []
            return inner
        module = types.SimpleNamespace(make_candidate=maker)
        class Tensor:
            def __init__(self, ptr, device='cuda:0'):
                self.ptr, self.device, self.dimensions = ptr, device, (16, 9216, 6144)
            def data_ptr(self): return self.ptr
            def stride(self, dim): return 6144 if self.ptr == 16 else 9216
        args = (Tensor(16), Tensor(32), Tensor(64), Tensor(80), Tensor(48))
        with mock.patch.dict(sys.modules, {'torch': torch_stub}), \
             mock.patch.object(candidate, '_validate', side_effect=lambda *xs: xs[1].dimensions), \
             mock.patch.object(candidate, '_load_snapshot', return_value=module):
            fn = candidate.make_candidate(build_dir=ROOT / 'build')
            capture[0] = True
            with self.assertRaisesRegex(RuntimeError, 'load/warm'):
                fn(*args)
            capture[0] = False
            self.assertEqual(fn(*args), 8)
            capture[0] = True
            self.assertEqual(fn(*args), 8)
            args[0].device = 'cuda:1'
            with self.assertRaisesRegex(RuntimeError, 'cfg/split'):
                fn(*args)
            capture[0] = False
            self.assertEqual(fn(*args), 8)
            self.assertEqual(len(made), 2)
            args[0].dimensions = (16, 9856, 6144)
            capture[0] = True
            with self.assertRaisesRegex(RuntimeError, 'cfg/split'):
                fn(*args)

    def test_makefile_preserves_compile_contract(self):
        text = (ROOT / 'Makefile').read_text()
        self.assertIn('-ffp-contract=off $<', text)
        self.assertIn('$(OPUS_INCLUDE_DIR)/../opus_gemm/include', text)
        commands = '\n'.join(line for line in text.splitlines() if line.startswith('\t'))
        self.assertNotIn('-ffast-math', commands)
        self.assertIn('gemm_a8w8_ptpc_splitk_kernel.cc', text)
        self.assertIn('gemm_a8w8_ptpc_8wave_kernel.cc', text)


if __name__ == '__main__':
    unittest.main()
