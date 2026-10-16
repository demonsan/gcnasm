# SPDX-License-Identifier: MIT
import argparse
import json
import statistics
from pathlib import Path

from policy import SHAPES, select

def shuffle_weight(weight):

    import torch
    n, k = weight.shape
    return weight.view(torch.uint8).reshape(n // 16, 16, k // 32, 2, 16).permute(
        0, 2, 3, 1, 4).contiguous().reshape(n, k).view(weight.dtype)

def reference(a, weight, sa, sb):

    return ((a.float() @ weight.float().t()) * sa.reshape(-1, 1) * sb.reshape(1, -1)).bfloat16()

def graph_time(fn, args, calls=101, samples=16):
    import torch
    nbytes = sum(t.numel() * t.element_size() for t in args)
    copies = min(calls, int(512 * 2**20 // nbytes) + 1)
    orbit = [args] + [tuple(t.clone() for t in args) for _ in range(copies - 1)]
    for item in orbit:
        fn(*item)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(*orbit[i % copies])
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record(); graph.replay(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return {'method': 'unprofiled_graph_preallocated', 'median_us': statistics.median(times),
            'min_us': min(times), 'max_us': max(times), 'raw_us': times,
            'calls_per_replay': calls, 'rotation_copies': copies}

def run_shape(torch, fn, m, n, k, seed, cfg, split, verify_only):
    torch.manual_seed(seed)
    a = (torch.randn(m, k, device='cuda') * 0.25).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, device='cuda') * 0.25).to(torch.float8_e4m3fn)
    sa = torch.rand(m, 1, device='cuda', dtype=torch.float32) * 0.25 + 0.5
    sb = torch.rand(n, 1, device='cuda', dtype=torch.float32) * 0.25 + 0.5
    wsh = shuffle_weight(w)
    ref = reference(a, w, sa, sb)

    storage = torch.full((m + 2, n), 7, device='cuda', dtype=torch.bfloat16)
    out = storage[1:-1]
    guards = storage[[0, -1]].clone()
    outputs = []
    for _ in range(3):
        for workspace in fn.active_ws:
            workspace.fill_(float('nan'))
        out.fill_(float('nan'))
        effective = fn(a, wsh, sa, sb, out)
        torch.cuda.synchronize()
        outputs.append(out.clone())
    finite = bool(torch.isfinite(out).all().item())
    close = bool(torch.isclose(out, ref, rtol=0.02, atol=0.02).all().item())
    repeat = all(torch.equal(x.view(torch.int16), outputs[0].view(torch.int16)) for x in outputs[1:])
    intact = torch.equal(storage[[0, -1]].view(torch.int16), guards.view(torch.int16))
    report = {'shape': [m, n, k], 'pass': bool(finite and close and repeat and intact),
              'ref_all_elements_0_02': close, 'finite': finite, 'repeat3_bits': repeat,
              'canaries': bool(intact), 'effective_split': effective,
              'derived': fn.derived, 'reported': fn.reported,
              'workspace_shapes': [list(t.shape) for t in fn.active_ws],
              'max_abs_diff': float((out.float() - ref.float()).abs().max().item())}
    if not report['pass']:
        raise RuntimeError(json.dumps(report))
    if not verify_only:
        report['timing'] = graph_time(fn, (a, wsh, sa, sb, out))
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', type=int, default=32)
    parser.add_argument('-n', type=int, default=6144)
    parser.add_argument('-k', type=int, default=8192)
    parser.add_argument('--cfg', choices=('c1', 'c11', 'c3', 'c18', 'c19'))
    parser.add_argument('--split', type=int, default=None, help='explicit split request, otherwise the example shape policy')
    parser.add_argument('--build-dir', default=str(Path(__file__).resolve().parent / 'build'))
    parser.add_argument('--matrix17', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='stdlib-only policy list; no Torch or GPU')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    shapes = SHAPES if args.matrix17 else ((args.m, args.n, args.k),)
    plan = [dict(shape=list(s), **select(*s, args.split, args.cfg).__dict__) for s in shapes]
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    import torch
    from candidate import make_candidate
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit('ROCm PyTorch and an admitted gfx950 GPU are required')
    print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
                      'arch': torch.cuda.get_device_properties(0).gcnArchName,
                      'scope': 'synthetic standalone smoke test; no CK/CKTile ranking or historical evidence transfer'}))
    for m, n, k in shapes:
        fn = make_candidate({'cfg': args.cfg}, split_k=args.split, build_dir=args.build_dir)
        print(json.dumps(run_shape(torch, fn, m, n, k, args.seed, args.cfg, args.split, args.verify_only)), flush=True)
        del fn
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
