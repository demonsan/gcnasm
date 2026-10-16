# SPDX-License-Identifier: MIT
from dataclasses import dataclass

NKS = ((9216, 6144), (9856, 6144), (6144, 8192))
SHAPES = tuple((m, n, k) for m in (16, 32, 2048, 4096, 8192) for n, k in NKS) + (
    (16, 7168, 8192), (2048, 7168, 8192),
)
M16_SPLITS = {(9216, 6144): 8, (9856, 6144): 4, (6144, 8192): 8, (7168, 8192): 4}
M32_SPLITS = {(9216, 6144): 3, (9856, 6144): 3, (6144, 8192): 4}
CONFIGS = {'c1': ('splitk', 16, 64, 256), 'c11': ('splitk', 32, 64, 256),
           'c3': ('eightwave', 128, 256, 128), 'c18': ('eightwave', 128, 256, 128),
           'c19': ('eightwave', 128, 256, 128)}

def ceil_div(a, b):
    return (a + b - 1) // b

@dataclass(frozen=True)
class Selection:
    artifact: str
    cfg: str
    requested_split: int
    bm: int
    bn: int
    bk: int

    def partition(self, k):
        tiles = ceil_div(k, self.bk)
        ips = ceil_div(tiles, self.requested_split)
        if self.artifact == 'eightwave' and tiles >= 2:
            ips += ips % 2
        return ips, ceil_div(tiles, ips)

    def workspace_shape(self, m, n, k):
        _, se = self.partition(k)
        return None if se == 1 else (se, ceil_div(m, self.bm) * self.bm,
                                    ceil_div(n, self.bn) * self.bn)

def select(m, n, k, split_k=None, cfg=None):
    if m < 1 or n < 16 or n % 16 or k < 32 or k % 32:
        raise ValueError('PTPC requires M>=1, N>=16 divisible by16, K>=32 divisible by32')
    if cfg is None:
        cfg = 'c1' if m <= 16 else ('c11' if m <= 32 else 'c19')
    if cfg not in CONFIGS:
        raise ValueError('Unsupported cfg; use c1, c11, c3, c18 or c19 (no c4 or alternate-scale-order entry)')
    artifact, bm, bn, bk = CONFIGS[cfg]
    if split_k is None:
        split_k = M16_SPLITS.get((n, k), 1) if m == 16 and cfg == 'c1' else (
            M32_SPLITS.get((n, k), 1) if m == 32 and cfg == 'c11' else 1)
    if not isinstance(split_k, int) or isinstance(split_k, bool) or not 1 <= split_k < 2**31:
        raise ValueError('requested split must be a positive signed-int32 integer')

    if ceil_div(k, bk) + split_k - 1 >= 2**31:
        raise ValueError('requested split exceeds signed-int32 partition headroom')
    return Selection(artifact, cfg, split_k, bm, bn, bk)

def describe(selection, m, n, k, a_ptr, w_ptr, c_ptr, lda, ldc):

    ips, se = selection.partition(k)
    ab = a_ptr % 16 == 0 and lda % 16 == 0 and w_ptr % 16 == 0
    ac = c_ptr % 16 == 0 and ldc % 8 == 0
    cfg = selection.cfg
    bm, bn, bk = selection.bm, selection.bn, selection.bk
    xcd = 0
    if cfg == 'c1':
        path = 'C1_MASK' if ab else 'C1_U'
    elif cfg == 'c11':
        tiles = ceil_div(k, 256)
        fast = ab and m % 32 == 0 and n % 64 == 0 and k % 256 == 0 and ips >= 3 and tiles - (se - 1) * ips >= 3
        path = 'C11_PC_A' if fast else ('C7_A' if ab else 'C2_U')
    else:
        target = cfg == 'c19' and ab and ac and selection.requested_split == 1 and (m, n, k) == (4096, 9216, 6144) and lda == k and ldc == n
        fast = cfg in ('c18', 'c19') and ab and ac and selection.requested_split == 1 and lda == k and ldc == n and k in (6144, 8192) and m >= 128
        if target:
            bm, bn, path, xcd = 256, 192, 'C19_256192', 4
        elif fast:
            bm, bn, path, xcd = 256, 256, 'C18_CONTRACT_ORDER', 4
        else:
            path = 'C3_U' if not ab else ('C3_EPI_A' if ac and se == 1 else 'C3_A')
    return {'cfg': cfg, 'requested_split': selection.requested_split, 's_eff': se,
            'iters_per_split': ips, 'pipeline': 'P4D' if selection.artifact == 'splitk' else 'P8',
            'tile': (bm, bn, bk), 'path': path, 'grid': ceil_div(m, bm) * ceil_div(n, bn) * se,
            'scale_order': '(acc*sa)*sb', 'xcd_swizzle': xcd, 'metadata_kind': 'input-derived selector, not device trace'}

def preshuffle_offset(n, k, K):
    return (n // 16) * 16 * K + (k // 32) * 512 + ((k % 32) // 16) * 256 + (n % 16) * 16 + k % 16
