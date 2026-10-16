# SPDX-License-Identifier: MIT
import importlib.util
import os
from pathlib import Path
import threading

from policy import describe, select

_LOAD_LOCK = threading.Lock()
_FILES = {'splitk': 'ptpc_c11_mature_gfx950.so', 'eightwave': 'ptpc_c19_256192_gfx950.so'}

def _load_snapshot(side, build_dir):

    build_dir = Path(build_dir).resolve()
    library = build_dir / _FILES[side]
    if not library.is_file():
        raise FileNotFoundError(f'{library}: build with make first')
    source = Path(__file__).resolve().parent / 'bindings' / (side + '_candidate.py')
    spec = importlib.util.spec_from_file_location(f'ptpc_snapshot_{side}_{id(build_dir)}', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _LOAD_LOCK:
        saved = {k: os.environ.get(k) for k in ('TASK_ROOT', 'SO_BASENAME')}
        os.environ['TASK_ROOT'] = str(build_dir.parent)

        os.environ['SO_BASENAME'] = str(library)
        try:
            module._load()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return module

def _validate(torch, xq, wsh, xs, ws, out):
    if any(t.ndim != 2 for t in (xq, wsh, out)):
        raise ValueError('A, preshuffled W and output must be rank2')
    m, k = xq.shape
    n, wk = wsh.shape
    if wk != k or tuple(out.shape) != (m, n):
        raise ValueError('inconsistent A/W/output shapes')
    if xq.dtype != torch.float8_e4m3fn or wsh.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
        raise ValueError('requires E4M3FN inputs and BF16 output')
    if xs.dtype != torch.float32 or ws.dtype != torch.float32:
        raise ValueError('PTPC scales must be FP32')
    if tuple(xs.shape) not in ((m,), (m, 1)) or tuple(ws.shape) not in ((n,), (n, 1)):
        raise ValueError('scales must be per-token [M] or [M,1], per-channel [N] or [N,1]')
    if not wsh.is_contiguous() or not xs.is_contiguous() or not ws.is_contiguous():
        raise ValueError('canonical W and scales must be contiguous')
    if xq.stride(1) != 1 or out.stride(1) != 1 or xq.stride(0) < k or out.stride(0) < n:
        raise ValueError('A/output require unit inner stride and non-overlapping row strides')
    tensors = (xq, wsh, xs, ws, out)
    if not all(t.is_cuda and t.device == xq.device for t in tensors):
        raise ValueError('all tensors must be on the same ROCm device')
    arch = getattr(torch.cuda.get_device_properties(xq.device), 'gcnArchName', '')
    if arch.split(':')[0] != 'gfx950':
        raise ValueError(f'this build supports gfx950 only, got {arch!r}')
    if max(m, n, k, xq.stride(0), out.stride(0)) >= 2**31:
        raise ValueError('dimensions/strides must fit signed int32')

    def span(t):
        elements = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        return t.data_ptr(), t.data_ptr() + elements * t.element_size()
    lo, hi = span(out)
    if any(lo < b and a < hi for a, b in map(span, tensors[:-1])):
        raise ValueError('output must not alias inputs or scales')

    return m, n, k

def make_candidate(d=None, split_k=None, build_dir=None):
    import torch
    options = dict(d or {})
    directory = Path(build_dir or options.get('build_dir') or Path(__file__).resolve().parent / 'build')
    modules, functions = {}, {}

    def launch(xq, wsh, xs, ws, out):
        m, n, k = _validate(torch, xq, wsh, xs, ws, out)
        requested = split_k if split_k is not None else options.get('split')
        if callable(requested):
            requested = requested(m, n, k)
        selection = select(m, n, k, requested, options.get('cfg'))
        side = selection.artifact
        if side not in modules:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('load/warm the selected artifact before graph capture')
            modules[side] = _load_snapshot(side, directory)

        key = (side, selection.cfg, selection.requested_split, str(xq.device))
        if key not in functions:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('prepare the selected cfg/split before graph capture')
            functions[key] = modules[side].make_candidate({'cfg': selection.cfg}, split_k=selection.requested_split)
        inner = functions[key]
        expected = describe(selection, m, n, k, xq.data_ptr(), wsh.data_ptr(), out.data_ptr(), xq.stride(0), out.stride(0))
        cache_key = key + (m, n, k)
        if cache_key not in run.prepared_shapes and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm this exact shape/split before graph capture (workspace allocation)')
        effective = inner(xq, wsh, xs, ws, out)
        if effective != expected['s_eff']:
            raise RuntimeError(f'returned split {effective} != derived {expected["s_eff"]}')
        run.prepared_shapes.add(cache_key)
        run.actual_cfg = inner.actual_cfg
        run.actual_artifact = side
        run.actual_s_eff = effective
        run.derived = expected
        run.reported = {field: getattr(inner, 'actual_' + field, None) for field in (
            'bm', 'bn', 'bk', 'path', 'grid_wgs', 'requested_split', 'scale_order', 'xcd_swizzle')}
        run.active_ws = inner.active_ws
        return effective

    def run(xq, wsh, xs, ws, out):

        with torch.cuda.device(xq.device):
            return launch(xq, wsh, xs, ws, out)

    run.active_ws = []
    run.prepared_shapes = set()
    run.actual_cfg = None
    return run
