"""Roundtrip + size test for lowlat.sparse.

Run:  python -m lowlat.test_sparse
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from lowlat.sparse import (
    METHOD_A_MASK,
    METHOD_B_ZLIB_CUR,
    METHOD_C_VARINT_ZLIB,
    decode_sparse_residual,
    encode_sparse_methods,
    encode_sparse_residual,
    encode_tile_auto,
)

H, W = 32, 32
NAMES = {METHOD_A_MASK: "A", METHOD_B_ZLIB_CUR: "B", METHOD_C_VARINT_ZLIB: "C"}


def make_pred(rng: np.random.Generator, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.int32)
    if kind == "gradient":
        base = np.stack([40 + xx, 60 + yy, 90 + (xx + yy) // 2], axis=2)
        return np.clip(base, 0, 255).astype(np.uint8)
    if kind == "flat":
        return np.full((H, W, 3), 128, dtype=np.uint8)
    if kind == "random":
        return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    raise ValueError(kind)


def make_resid(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "sparse95":
        mask = rng.random((H, W, 3)) < 0.05  # exactly ~95% zeros
        vals = rng.integers(-255, 256, (H, W, 3), dtype=np.int16)
        return np.where(mask, vals, 0).astype(np.int16)
    if kind == "text_edges":
        r = np.zeros((H, W, 3), dtype=np.int16)
        for _ in range(60):  # short text-like strokes
            y = int(rng.integers(0, H - 2))
            x = int(rng.integers(0, W - 12))
            ln = int(rng.integers(3, 12))
            r[y : y + 2, x : x + ln, :] = rng.integers(-60, 61, (2, ln, 3), dtype=np.int16)
        for _ in range(4):  # a few full-width rules
            y = int(rng.integers(0, H))
            r[y, :, :] = rng.integers(-30, 31, (W, 3), dtype=np.int16)
        return r
    if kind == "flat_noise":
        return np.clip(np.rint(rng.normal(0, 8, (H, W, 3))), -255, 255).astype(np.int16)
    if kind == "all_zero":
        return np.zeros((H, W, 3), dtype=np.int16)
    raise ValueError(kind)


def check_case(name: str, resid: np.ndarray, pred: np.ndarray) -> dict:
    cur = np.clip(pred.astype(np.int16) + resid, 0, 255).astype(np.uint8)
    methods = encode_sparse_methods(resid, pred)
    sizes = {mid: len(p) for mid, p in methods.items()}

    # per-method roundtrip (each method in its own container)
    ok = True
    for mid, payload in methods.items():
        direct = struct.pack("<BI", mid, len(payload)) + payload
        ok &= bool(np.array_equal(decode_sparse_residual(direct, pred), cur))

    # auto container + encode_tile_auto
    auto = encode_sparse_residual(resid, pred)
    ok &= bool(np.array_equal(decode_sparse_residual(auto, pred), cur))
    auto2 = encode_tile_auto(cur, pred)
    ok &= bool(np.array_equal(decode_sparse_residual(auto2, pred), cur))
    chosen = auto[0]
    naive = len(zlib.compress(resid.tobytes(), 6))
    return {
        "name": name,
        "sizes": sizes,
        "chosen": chosen,
        "chosen_size": len(auto),
        "naive": naive,
        "ok": ok,
        "auto": auto,
    }


def main() -> int:
    rng = np.random.default_rng(0)
    cases = [
        ("sparse95/gradient", "sparse95", "gradient"),
        ("sparse95/randompred", "sparse95", "random"),
        ("text_edges/gradient", "text_edges", "gradient"),
        ("flat_noise/flat", "flat_noise", "flat"),
        ("all_zero/gradient", "all_zero", "gradient"),
    ]

    results = []
    for name, rk, pk in cases:
        resid = make_resid(rng, rk)
        pred = make_pred(rng, pk)
        results.append(check_case(name, resid, pred))

    hdr = f"{'case':24s} {'naive':>7s} {'A':>7s} {'B':>7s} {'C':>7s} {'chosen':>9s}  roundtrip"
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    sparse_ok = True
    for r in results:
        s = r["sizes"]
        chosen = f"{NAMES[r['chosen']]}={r['chosen_size']}"
        all_ok &= r["ok"]
        if r["name"].startswith("sparse95"):
            sparse_ok &= r["chosen_size"] < r["naive"]
        print(
            f"{r['name']:24s} {r['naive']:7d} {s[METHOD_A_MASK]:7d} "
            f"{s[METHOD_B_ZLIB_CUR]:7d} {s[METHOD_C_VARINT_ZLIB]:7d} "
            f"{chosen:>9s}  {'OK' if r['ok'] else 'FAIL'}"
        )

    print()
    print(f"roundtrip bit-exact (all cases, all methods): {'PASS' if all_ok else 'FAIL'}")
    print(
        "target: sparse95 container beats naive zlib(full residual): "
        f"{'PASS' if sparse_ok else 'FAIL'}"
    )
    # sparsity sanity of the sparse case
    resid = make_resid(np.random.default_rng(123), "sparse95")
    frac = float(np.mean(resid == 0))
    print(f"sparse95 zero fraction: {frac:.4f} (target ~0.95)")
    return 0 if (all_ok and sparse_ok and frac > 0.9) else 1


if __name__ == "__main__":
    raise SystemExit(main())
