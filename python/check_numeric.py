"""Runnable boundary checks and full-int8-domain bounds for the accepted model."""
import numpy as np
from engine import (ONE, SCALE_SHIFT, LUT_OFF, QLayer, _check_bounds,
                    _apply_scale_bias, quantize_layer, dense, tconv, conv1x1,
                    lut_lookup, build_tanh_lut, render_rgb)
from export import load_qmodel


def model_bounds(m):
    # Absolute bounds per input channel; all calculations below are Python ints.
    bounds = [128 * ONE] * 32
    report = []
    for name, layer in (("fc", m.fc), ("t0", m.t0), ("t1", m.t1),
                        ("t2", m.t2), ("out", m.out)):
        w = layer.wq.astype(object)
        if name.startswith("t"):
            # Each output location receives only one kernel parity (at most 2x2).
            rows = [[sum(abs(int(w[i, o, kh, kw])) * bounds[i]
                         for i in range(len(bounds))
                         for kh in range(ph, 4, 2) for kw in range(pw, 4, 2))
                     for ph in range(2) for pw in range(2)]
                    for o in range(w.shape[1])]
            accum = [max(row) for row in rows]
        else:
            accum = [sum(abs(int(v)) * b for v, b in zip(row, bounds)) for row in w]
        bounds = []
        products = []
        for a, s, b in zip(accum, layer.scale, layer.bias):
            _check_bounds(-a, a)
            p = a * int(s)
            _check_bounds(-p, p)
            lo, hi = (-p >> SCALE_SHIFT) + int(b), (p >> SCALE_SHIFT) + int(b)
            _check_bounds(lo, hi)
            bounds.append(max(abs(lo), abs(hi)) if name == "out" else max(0, hi))
            products.append(p)
        assert max(accum) < 2**62, f"{name}: signed accumulator lane overflow"
        assert max(bounds) < 2**63, f"{name}: activation lane overflow"
        report.append((name, max(accum), max(products), max(bounds)))
        if name == "fc":
            bounds = [max(bounds[i:i + 16]) for i in range(0, len(bounds), 16)]
    _check_bounds(-max(bounds) + LUT_OFF, max(bounds) + LUT_OFF)
    return report


def rejects(fn):
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError("unsafe input was accepted")


def main():
    for bad in (np.nan, np.inf, -np.inf):
        rejects(lambda: quantize_layer([[bad]], [0], 0))
        rejects(lambda: quantize_layer([[1]], [bad], 0))
    for b in (2**19, -2**19 - 1):
        rejects(lambda: quantize_layer([[1]], [b], 0))
    assert quantize_layer([[1]], [-2**19], 0).bias[0] == -2**31
    rejects(lambda: quantize_layer([[128]], [0], 0))
    for a, s, b in ((2**62, 4, 0), (-2**63, 2, 0), (-2**63, -1, 0),
                    (1, 1, 2**63 - 1), (-2**32, 1, -2**63)):
        if a == 1:
            a = 2**32
        rejects(lambda: _apply_scale_bias(np.array([a]), np.array([s]), np.array([b]), False))
    assert _apply_scale_bias(np.array([-1]), np.array([1]), np.array([0]), False)[0] == -1
    rejects(lambda: _apply_scale_bias(np.array([0, 2**62]), np.array([4]), np.array([0]), False))
    rejects(lambda: dense(np.array([2**62]), QLayer(np.array([[4]]), np.array([1]), np.array([0]))))
    rejects(lambda: tconv(np.full((1, 1, 1), 2**62), QLayer(np.ones((1, 1, 4, 4), dtype=int), np.array([1]), np.array([0]))))
    rejects(lambda: conv1x1(np.full((1, 1, 1), 2**62), QLayer(np.array([[4]]), np.array([1]), np.array([0]))))
    rejects(lambda: lut_lookup(np.array([2**63 - 1]), build_tanh_lut()))
    rejects(lambda: render_rgb(None, [128] * 32))
    lut = build_tanh_lut()
    pre = np.array([-32769, -32768, -32737, -32736, 32735, 32736, 32767, 32768])
    assert np.array_equal(lut_lookup(pre, lut), lut[[0, 0, 0, 1, 2046, 2047, 2047, 2047]])
    m = load_qmodel()
    print("layer   max |acc|    max |acc*scale|    max |activation|")
    for row in model_bounds(m):
        print(*row)
        print(f"  lane headroom: accumulator {62 - row[1].bit_length()} bits, "
              f"activation {63 - row[3].bit_length()} bits")
    print("PASS: numeric boundary checks and full int8 latent domain fit int64")


if __name__ == "__main__":
    main()
