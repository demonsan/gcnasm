# SPDX-License-Identifier: MIT
import ctypes
import os

import torch

_LIB = None
_CFG = {
    "c19": ("ptpc_c19_launch", 128, 256),
    "c18": ("ptpc_c18_launch", 128, 256),
    "c3": ("ptpc_c3_launch", 128, 256),
}
_ARGTYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

def _load():
    global _LIB
    if _LIB is None:
        task_root = os.environ.get("TASK_ROOT", os.path.dirname(os.path.dirname(__file__)))
        so_name = os.environ.get("SO_BASENAME", "ptpc_c19_256192_gfx950.so")
        _LIB = ctypes.CDLL(os.path.join(task_root, "build", so_name))
        for name, _, _ in _CFG.values():
            fn = getattr(_LIB, name)
            fn.restype = ctypes.c_int
            fn.argtypes = _ARGTYPES
        _LIB.ptpc_last_hip_error.restype = ctypes.c_int
        _LIB.ptpc_last_hip_error.argtypes = []
    return _LIB

def _default_cfg(m):
    return "c19"

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
        s_arg=s
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
                out.data_ptr(), ws_ptr, ws_bytes, m, n, k, stride_a, stride_c, s_arg,
                ctypes.byref(s_eff), stream)
        if rc != 0:
            raise RuntimeError(f"ptpc launch rc={rc} (m={m}, n={n}, k={k}, split={s}, cfg={cfg})")
        hip_rc = lib.ptpc_last_hip_error()
        if hip_rc != 0:
            raise RuntimeError(f"ptpc host HIP error={hip_rc} (m={m}, n={n}, k={k}, split={s}, cfg={cfg})")
        if cfg in ("c18","c19"):
            ab=xq.data_ptr()%16==0 and stride_a%16==0 and wsh.data_ptr()%16==0
            ac=out.data_ptr()%16==0 and stride_c%8==0
            fast=ab and ac and s_arg==1 and stride_a==k and stride_c==n and k in(6144,8192) and m>=128
            candidate.actual_bm=256 if fast else 128;candidate.actual_bn=256;candidate.actual_bk=128
            candidate.actual_path="C18_CONTRACT_ORDER"if fast else("C3_EPI_A"if ab and ac and s_eff.value==1 else("C3_A"if ab else"C3_U"))
            candidate.actual_grid_wgs=((m+255)//256)*((n+255)//256)if fast else((m+127)//128)*((n+255)//256)*s_eff.value
            candidate.actual_requested_split=s_arg
            candidate.actual_scale_order="(acc*sa)*sb"
            candidate.actual_xcd_swizzle=4 if fast else 0
        if cfg=="c19" and fast and (m,n,k)==(4096,9216,6144):
            candidate.actual_bm=256;candidate.actual_bn=192;candidate.actual_bk=128
            candidate.actual_path="C19_256192";candidate.actual_grid_wgs=768
            candidate.actual_scale_order="(acc*sa)*sb";candidate.actual_xcd_swizzle=4
        return s_eff.value

    candidate.active_ws = active_ws
    candidate.actual_cfg = cfg_req
    return candidate
