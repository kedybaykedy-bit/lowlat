#!/usr/bin/env python3
"""Tests + benchmark for lowlat/freq (C largest-remainder quantization).

Run: python3 lowlat/test_freq.py
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lowlat import freq  # noqa: E402

TOTAL = 65536
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))


def reference_freqs(probs: np.ndarray) -> np.ndarray:
    """Simple numpy largest-remainder reference (float32, mirrors freq.c)."""
    flat = np.ascontiguousarray(probs, dtype=np.float32)
    assert flat.ndim == 2
    n_rows, n_sym = flat.shape
    p = np.where(flat >= 0, flat, np.float32(0))
    target = TOTAL - n_sym
    ftarget = np.float32(target)
    scaled = p * ftarget
    np.minimum(scaled, ftarget, out=scaled)
    fl = np.floor(scaled)
    base = fl.astype(np.int64) + 1
    frac = scaled - fl
    rem = TOTAL - base.sum(axis=1)
    order = np.argsort(frac, axis=1, kind="stable")
    out = base
    for i in range(n_rows):
        r = int(rem[i])
        if r > 0:
            cycles, r = divmod(r, n_sym)
            if cycles:
                out[i] += cycles
            if r:
                out[i, order[i, n_sym - r :]] += 1
        elif r < 0:
            need = -r
            while need > 0:
                moved = False
                for k in range(n_sym):
                    if need == 0:
                        break
                    j = int(order[i, k])
                    if out[i, j] > 1:
                        out[i, j] -= 1
                        need -= 1
                        moved = True
                if not moved:
                    break
    return out.astype(np.uint32)


def load_codec_freqs():
    """Extract codec.py::_freqs_from_logits without importing torch."""
    path = HERE / "codec.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_freqs_from_logits":
                    item.decorator_list = []
                    mod = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    ns = {"np": np}
                    exec(compile(mod, str(path), "exec"), ns)
                    return ns["_freqs_from_logits"]
    raise RuntimeError("codec._freqs_from_logits not found")


def softmax32(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def bench(fn, x, n: int = 30) -> float:
    fn(x)  # warmup
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"libfreq: HAVE_C={freq.HAVE_C}  lib={freq._LIB}")

    # ---- 1. random rows: all sums == 65536 ----
    print("\n[1] sums == 65536")
    logits = rng.standard_normal((32, 32, 3, 511)).astype(np.float32)
    f = freq.freqs_from_logits(logits)
    sums = f.astype(np.uint64).sum(axis=-1)
    check("random (32,32,3,511) sums", bool(np.all(sums == TOTAL)),
          f"shape={f.shape} dtype={f.dtype} min={sums.min()} max={sums.max()}")
    check("dtype/shape preserved", f.dtype == np.uint32 and f.shape == logits.shape)
    lg0 = logits.copy()
    freq.freqs_from_logits(logits)
    check("input logits unmodified", np.array_equal(logits, lg0))
    check("min freq >= 1", bool(f.min() >= 1), f"min={f.min()}")

    # degenerate / edge shapes
    for shape, nsym in [((7, 1), 1), ((5, 17), 17), ((3, 65536), 65536)]:
        lg = rng.standard_normal(shape).astype(np.float32)
        ff = freq.freqs_from_logits(lg)
        s = ff.astype(np.uint64).sum(axis=-1)
        check(f"edge sums {shape}", bool(np.all(s == TOTAL)), f"min={s.min()} max={s.max()}")

    # ---- 2. match numpy largest-remainder reference ----
    print("\n[2] vs numpy largest-remainder reference")
    exact_all = True
    worst = 0
    for n_sym in (1, 2, 17, 64, 511, 1024, 4096):
        lg = rng.standard_normal((64, n_sym)).astype(np.float32)
        probs = softmax32(lg)
        got = freq.freqs_from_probs(probs)
        exp = reference_freqs(probs)
        exact = np.array_equal(got, exp)
        exact_all &= exact
        if not exact:
            d = np.abs(got.astype(np.int64) - exp.astype(np.int64)).max()
            worst = max(worst, int(d))
        sums = got.astype(np.uint64).sum(axis=-1)
        check(f"n_sym={n_sym}", exact and bool(np.all(sums == TOTAL)),
              "" if exact else f"max_abs_diff={int(np.abs(got.astype(np.int64)-exp.astype(np.int64)).max())}")

    # structured / adversarial probability rows
    cases = {}
    cases["uniform"] = np.full((4, 511), np.float32(1.0 / 511), dtype=np.float32)
    onehot = np.zeros((3, 511), dtype=np.float32)
    onehot[:, 0] = 1.0
    cases["one-hot"] = onehot
    cases["zeros"] = np.zeros((2, 511), dtype=np.float32)
    ext = np.full((3, 511), np.float32(-80), dtype=np.float32)
    ext[:, 100] = np.float32(80)
    cases["extreme"] = softmax32(ext)
    # rows not summing to exactly 1
    u = rng.random((4, 511)).astype(np.float32)
    cases["rowsum_0.9"] = u * (np.float32(0.9) / u.sum(axis=1, keepdims=True))
    cases["rowsum_1.1"] = u * (np.float32(1.1) / u.sum(axis=1, keepdims=True))
    bad = u.copy()
    bad[0, 0] = np.nan
    bad[1, 5] = np.float32(-0.5)
    cases["nan_neg"] = bad
    for name, probs in cases.items():
        got = freq.freqs_from_probs(probs)
        exp = reference_freqs(probs)
        exact = np.array_equal(got, exp)
        exact_all &= exact
        s = got.astype(np.uint64).sum(axis=-1)
        check(f"case {name}", exact and bool(np.all(s == TOTAL)),
              "" if exact else f"max_abs_diff={int(np.abs(got.astype(np.int64)-exp.astype(np.int64)).max())} sums_ok={bool(np.all(s==TOTAL))}")

    # numpy fallback path must match C path
    if freq.HAVE_C:
        probs = softmax32(rng.standard_normal((128, 511)).astype(np.float32))
        check("fallback numpy == C", np.array_equal(freq._quantize_numpy(probs),
                                                    freq.freqs_from_probs(probs)))

    # full pipeline: freqs_from_logits == reference(softmax(logits))
    lg = rng.standard_normal((32, 32, 3, 511)).astype(np.float32)
    got = freq.freqs_from_logits(lg)
    exp = reference_freqs(softmax32(lg).reshape(-1, 511)).reshape(lg.shape)
    check("pipeline vs reference", np.array_equal(got, exp))

    # ---- 3. benchmark: 32x32x3 rows x 511 sym ----
    print("\n[3] benchmark (32,32,3,511) = 3072 rows x 511 sym")
    codec_ref = load_codec_freqs()
    lg = rng.standard_normal((32, 32, 3, 511)).astype(np.float32)
    probs = softmax32(lg)

    t_codec = bench(codec_ref, lg)
    t_c = bench(freq.freqs_from_logits, lg)
    t_qc = bench(freq.freqs_from_probs, probs)
    flat = probs.reshape(-1, 511)
    t_qn = bench(freq._quantize_numpy, flat)
    print(f"  codec._freqs_from_logits (numpy, argmax dump) : {t_codec:8.3f} ms")
    print(f"  freq.freqs_from_logits  (softmax + C quant)    : {t_c:8.3f} ms")
    print(f"    speedup vs codec.py                          : {t_codec / t_c:8.2f}x")
    print(f"  quantize-only  C                              : {t_qc:8.3f} ms")
    print(f"  quantize-only  numpy largest-remainder         : {t_qn:8.3f} ms")
    print(f"    quantize speedup                             : {t_qn / t_qc:8.2f}x")

    # ---- summary ----
    npass = sum(1 for _, ok, _ in _results if ok)
    nfail = len(_results) - npass
    print(f"\n{npass}/{len(_results)} passed, {nfail} failed"
          + ("" if exact_all else "  (reference mismatch present)"))
    print(f".so: {freq._LIB}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
