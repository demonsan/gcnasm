# SPDX-License-Identifier: MIT
import ctypes
import os

import torch

_LIB = None
_CFG = {
    "c1": ("ptpc_c1_launch", 16, 64),
    "c11": ("ptpc_c11_launch", 32, 64),
}
_ARGTYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

def _load():
    global _LIB
    if _LIB is None:
        task_root = os.environ.get("TASK_ROOT", os.path.dirname(os.path.dirname(__file__)))
        so_name = os.environ.get("SO_BASENAME", "ptpc_c11_mature_gfx950.so")
        _LIB = ctypes.CDLL(os.path.join(task_root, "build", so_name))
        for name, _, _ in _CFG.values():
            fn = getattr(_LIB, name)
            fn.restype = ctypes.c_int
            fn.argtypes = _ARGTYPES
        _LIB.ptpc_last_hip_error.restype = ctypes.c_int
        _LIB.ptpc_last_hip_error.argtypes = []
    return _LIB

def _default_cfg(m):
    return "c1" if m <= 16 else "c11"

def make_candidate(d, split_k=None):
    lib = _load()
    cfg_req = d.get("cfg")
    split_sel = split_k if split_k is not None else d.get("split", 1)
    cache = {}
    fns = {name: getattr(lib, name) for name, _, _ in _CFG.values()}
    active_ws = []

    def candidate(xq, wsh, xs, ws, out):
        m, k = int(xq.shape[-2]), int(xq.shape[-1])
        n = int(wsh.shape[-2])
        cfg = cfg_req or _default_cfg(m)
        fn_name, bm, bn = _CFG[cfg]
        fn = fns[fn_name]
        candidate.actual_cfg = cfg
        stride_a = int(xq.stride(-2))
        stride_c = int(out.stride(-2))
        s = int(split_sel(m, n, k)) if callable(split_sel) else int(split_sel)
        stream = torch.cuda.current_stream().cuda_stream
        key = (m, n, k, s)
        wsp = cache.get(key, False)
        if wsp is False:
            if s <= 1:
                wsp = None
            else:
                tiles_m = -(-m // bm)
                tiles_n = -(-n // bn)
                wsp = torch.full((s, tiles_m * bm, tiles_n * bn), float("nan"),
                                 dtype=torch.float32, device=xq.device)
                active_ws.append(wsp)
            cache[key] = wsp
        ws_ptr = wsp.data_ptr() if wsp is not None else None
        ws_bytes = wsp.numel() * 4 if wsp is not None else 0
        s_eff = ctypes.c_int(-1)
        rc = fn(xq.data_ptr(), wsh.data_ptr(), xs.data_ptr(), ws.data_ptr(),
                out.data_ptr(), ws_ptr, ws_bytes, m, n, k, stride_a, stride_c, s,
                ctypes.byref(s_eff), stream)
        if rc != 0:
            raise RuntimeError(f"ptpc launch rc={rc} (m={m}, n={n}, k={k}, split={s}, cfg={cfg})")
        hip_rc = lib.ptpc_last_hip_error()
        if hip_rc != 0:
            raise RuntimeError(f"ptpc host HIP error={hip_rc} (m={m}, n={n}, k={k}, split={s}, cfg={cfg})")

        if cfg == "c11":
            aligned = xq.data_ptr()%16==0 and stride_a%16==0 and wsh.data_ptr()%16==0
            t=(k+255)//256; ips=(t+s-1)//s; actual_se=(t+ips-1)//ips
            fast=aligned and m%32==0 and n%64==0 and k%256==0 and ips>=3 and t-(actual_se-1)*ips>=3
            candidate.actual_bm=32;candidate.actual_bn=64;candidate.actual_bk=256
            candidate.actual_path="C11_PC_A" if fast else ("C7_A" if aligned else "C2_U")
        return s_eff.value

    candidate.active_ws = active_ws
    candidate.actual_cfg = cfg_req
    return candidate
