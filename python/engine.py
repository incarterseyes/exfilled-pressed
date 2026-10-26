"""Per-output-channel fixed-point inference engine — the byte-exact Python twin
of Pressed's on-chain renderer.

Compared with a plain int8 engine it adds:
  * one per-OUTPUT-CHANNEL scale (uint32, Q0.SCALE_SHIFT) per weight tensor,
  * biases carried in Q(ACT_SHIFT) at their TRUE scale (not int8 * 128),
  * a tanh->pixel LUT for the output stage (not a linear affine).

Fixed-point conventions (identical to the contract):
  activations  x are integers in Q(ACT_SHIFT):  x_int = round(x_real * 2^ACT_SHIFT)
  a weight row o is quantized  wq[o] = round(w_real[o] / s_o),  s_o = max|w_real[o]|/127
  scale_q[o]   = round(s_o * 2^SCALE_SHIFT)              (uint32)
  bias_q[o]    = round(bias_real[o] * 2^ACT_SHIFT)       (int32)
  layer out    = act( (acc * scale_q[o]) >> SCALE_SHIFT + bias_q[o] )   in Q(ACT_SHIFT)
                 where acc = Σ wq[o,i] * x_int[i]
Hidden layers use ReLU; the output conv is linear, then the tanh LUT maps its
Q(ACT_SHIFT) value to a 0..255 pixel.

All arithmetic is exact integer math (EVM is 256-bit); numpy int64 is checked to
not overflow for this model size.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ACT_SHIFT = 12
ONE = 1 << ACT_SHIFT           # Q12.12 unit for activations
SCALE_SHIFT = 32               # per-channel weight scale is Q0.32 (uint32)
IMG = 32
RGB_BYTES = IMG * IMG * 3

# tanh LUT: index = clamp((v + LUT_OFF) >> LUT_RSH, 0, LUT_N-1), v in Q(ACT_SHIFT).
# 2048 bins of 32 (Q12.12) cover v in [-32768, 32768) i.e. real [-8, 8); tanh is
# fully saturated past |8|, so this is the same bin resolution as a wider table
# at half the bytes — chosen so the whole payload fits one <24576-byte contract.
LUT_N = 2048
LUT_RSH = 5                    # 32 = 2^5 per bin
LUT_OFF = (LUT_N << LUT_RSH) // 2   # 32768


def _q_i64(a):
    a = np.asarray(a)
    if a.dtype.kind not in "iu" or (a.size and
            (int(a.min()) < -(1 << 63) or int(a.max()) >= (1 << 63))):
        raise ValueError("expected integers in int64 range")
    return a.astype(np.int64)


def _check_bounds(lo, hi):
    if lo < -(1 << 63) or hi >= (1 << 63):
        raise OverflowError("integer operation would overflow int64")


def _acc_bound(x, w, terms):
    # Python integers: abs(INT64_MIN) and the products cannot wrap.
    x, w = _q_i64(x), _q_i64(w)
    bound = max(abs(int(x.min())), abs(int(x.max()))) * max(
        abs(int(w.min())), abs(int(w.max()))) * terms
    _check_bounds(-bound, bound)


@dataclass
class QLayer:
    wq: np.ndarray        # int8, shape (out, in[, kh, kw])
    scale: np.ndarray     # uint32, shape (out,)
    bias: np.ndarray      # int32,  shape (out,)


def quantize_layer(w_real: np.ndarray, b_real: np.ndarray, out_axis: int) -> QLayer:
    """Per-output-channel symmetric int8 weight quant + Q-scaled bias."""
    w = np.asarray(w_real, dtype=np.float64)
    b = np.asarray(b_real, dtype=np.float64)
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("weights and biases must be finite")
    dims = tuple(d for d in range(w.ndim) if d != out_axis)
    amax = np.maximum(np.abs(w).max(axis=dims, keepdims=True), 1e-12)
    s = amax / 127.0                                  # per-out-channel float scale
    wq = np.round(w / s)
    if not np.isfinite(wq).all() or wq.min() < -127 or wq.max() > 127:
        raise ValueError("quantized weight outside int8 range")
    wq = wq.astype(np.int8)
    scale_q = np.round(s.reshape(-1) * (1 << SCALE_SHIFT))
    if not np.isfinite(scale_q).all() or scale_q.min() < 0 or scale_q.max() >= (1 << 32):
        raise ValueError(f"per-channel scale overflows uint32 (max {scale_q.max()}); "
                         "raise SCALE_SHIFT headroom or widen to uint64")
    scale_q = scale_q.astype(np.uint32)
    bias_q = np.round(b * ONE)
    if not np.isfinite(bias_q).all() or bias_q.min() < -(1 << 31) or bias_q.max() >= (1 << 31):
        raise ValueError("quantized bias outside int32 range")
    bias_q = bias_q.astype(np.int32)
    return QLayer(wq=wq, scale=scale_q, bias=bias_q)


def _apply_scale_bias(acc: np.ndarray, scale: np.ndarray, bias: np.ndarray,
                      relu: bool) -> np.ndarray:
    """(acc * scale) >> SCALE_SHIFT + bias, arithmetic shift; optional ReLU.
    acc shape (out, ...); scale/bias shape (out,). Returns int64 Q(ACT_SHIFT)."""
    acc = _q_i64(acc)
    scale, bias = _q_i64(scale), _q_i64(bias)
    if acc.ndim == 0 or scale.shape != (acc.shape[0],) or bias.shape != scale.shape:
        raise ValueError("scale and bias must have one value per output channel")
    for a, s, b in zip(acc, scale, bias):
        ends = [int(a.min()) * int(s), int(a.max()) * int(s)]
        lo, hi = min(ends), max(ends)
        _check_bounds(lo, hi)
        _check_bounds((lo >> SCALE_SHIFT) + int(b), (hi >> SCALE_SHIFT) + int(b))
    prod = acc * scale.reshape((-1,) + (1,) * (acc.ndim - 1))
    v = (prod >> SCALE_SHIFT) + bias.reshape((-1,) + (1,) * (acc.ndim - 1))
    if relu:
        np.maximum(v, 0, out=v)
    return v


def dense(x: np.ndarray, L: QLayer, relu: bool = True) -> np.ndarray:
    _acc_bound(x, L.wq, L.wq.shape[1])
    acc = _q_i64(L.wq.astype(np.int64)) @ _q_i64(x)          # (out,)
    return _apply_scale_bias(acc, L.scale, L.bias, relu)


def tconv(x: np.ndarray, L: QLayer, relu: bool = True) -> np.ndarray:
    """Transposed conv, 4x4 kernel stride 2 pad 1 (oh = 2*ih + kh - 1).
    x: (in, H, W) int64; L.wq: (in, out, 4, 4)."""
    in_ch, H, W = x.shape
    x = _q_i64(x)
    _acc_bound(x, L.wq, in_ch * 4)  # four spatial contributions per output/channel
    out_ch = L.wq.shape[1]
    oH, oW = H * 2, W * 2
    acc = np.zeros((out_ch, oH, oW), dtype=np.int64)
    w = L.wq.astype(np.int64)
    for kh in range(4):
        off_h = kh - 1
        ih_lo = max(0, (-off_h + 1) // 2); ih_hi = min(H - 1, (oH - 1 - off_h) // 2)
        if ih_lo > ih_hi:
            continue
        for kw in range(4):
            off_w = kw - 1
            iw_lo = max(0, (-off_w + 1) // 2); iw_hi = min(W - 1, (oW - 1 - off_w) // 2)
            if iw_lo > iw_hi:
                continue
            xs = x[:, ih_lo:ih_hi + 1, iw_lo:iw_hi + 1]
            contrib = np.einsum("io,ihw->ohw", w[:, :, kh, kw], xs)
            acc[:, off_h + 2 * ih_lo:off_h + 2 * ih_hi + 1:2,
                   off_w + 2 * iw_lo:off_w + 2 * iw_hi + 1:2] += contrib
    return _apply_scale_bias(acc, L.scale, L.bias, relu)


def conv1x1(x: np.ndarray, L: QLayer, relu: bool = False) -> np.ndarray:
    """1x1 conv: L.wq (out, in). x: (in, H, W)."""
    in_ch, H, W = x.shape
    _acc_bound(x, L.wq, in_ch)
    acc = np.einsum("oi,ihw->ohw", L.wq.astype(np.int64), _q_i64(x))
    return _apply_scale_bias(acc, L.scale, L.bias, relu)


def build_tanh_lut() -> np.ndarray:
    """LUT_N entries: pre (Q(ACT_SHIFT)) -> pixel 0..255 via (tanh(pre)+1)*127.5."""
    idx = np.arange(LUT_N)
    pre = (idx << LUT_RSH) - LUT_OFF          # center of each bin, Q(ACT_SHIFT)
    real = pre.astype(np.float64) / ONE
    px = np.round((np.tanh(real) + 1.0) * 127.5)
    return np.clip(px, 0, 255).astype(np.uint8)


def lut_lookup(pre: np.ndarray, lut: np.ndarray) -> np.ndarray:
    pre = _q_i64(pre)
    _check_bounds(int(pre.min()) + LUT_OFF, int(pre.max()) + LUT_OFF)
    idx = (pre + LUT_OFF) >> LUT_RSH
    idx = np.clip(idx, 0, LUT_N - 1)
    return lut[idx]


@dataclass
class QModel:
    fc: QLayer
    t0: QLayer
    t1: QLayer
    t2: QLayer
    out: QLayer
    lut: np.ndarray
    c0: int


def render_rgb(model: QModel, latent: list) -> bytes:
    """latent (32 int8) -> 3072 interleaved RGB bytes (32x32)."""
    x = _q_i64(latent)
    if x.shape != (32,) or x.min() < -128 or x.max() > 127:
        raise ValueError("latent must contain 32 int8 values")
    x = x * ONE                                    # Q(ACT_SHIFT)
    h = dense(x, model.fc, relu=True).reshape(model.c0, 4, 4)
    h = tconv(h, model.t0, relu=True)
    h = tconv(h, model.t1, relu=True)
    h = tconv(h, model.t2, relu=True)
    pre = conv1x1(h, model.out, relu=False)          # (3,32,32) Q(ACT_SHIFT)
    px = lut_lookup(pre, model.lut)                  # (3,32,32) uint8
    return px.transpose(1, 2, 0).tobytes()
