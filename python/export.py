"""Checkpoint -> quantized QModel + the packed weight payload in weights/blobs.

Blob byte layouts (big-endian; the same layout Pressed stores on-chain):
  <LAYER>_W   int8 weights, OUTPUT-CHANNEL major:
                fc / out : [o][i]
                tconv    : [o][i][kh][kw]      (4x4 kernels)
  <LAYER>_S   uint32 per output channel  (Q0.SCALE_SHIFT)   4 bytes each
  <LAYER>_B   int32  per output channel  (Q ACT_SHIFT)      4 bytes each
  TANH_LUT    LUT_N uint8 pixels
The sections are concatenated into one payload; renderers read each section by
the offsets recorded in manifest.json.
"""
from __future__ import annotations
import json, os, hashlib
import numpy as np

from engine import (QLayer, QModel, quantize_layer, build_tanh_lut,
                    ACT_SHIFT, SCALE_SHIFT, LUT_N, LUT_RSH, LUT_OFF)

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "weights", "oxford_mid20_step30000_g_ema.pt")
OUTDIR = os.path.join(HERE, "..", "weights", "blobs")

# torch weight tensor -> output-channel-major int8 bytes
def _w_bytes(wq: np.ndarray, kind: str) -> bytes:
    if kind == "linear":                 # (out, in)
        return wq.astype(np.int8).tobytes()
    if kind == "tconv":                  # torch (in, out, 4, 4) -> (out, in, 4, 4)
        return np.transpose(wq, (1, 0, 2, 3)).astype(np.int8).tobytes()
    raise ValueError(kind)

def _u32(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=">u4").tobytes()

def _i32(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=">i4").tobytes()


def load_qmodel(checkpoint: str = CKPT) -> QModel:
    import torch
    from model32 import C0, C3
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("G_ema", ck)
    npf = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in sd.items()}
    fc = quantize_layer(npf["fc.weight"], npf["b_fc"], out_axis=0)
    t0 = quantize_layer(npf["t0.weight"], npf["b0"], out_axis=1)
    t1 = quantize_layer(npf["t1.weight"], npf["b1"], out_axis=1)
    t2 = quantize_layer(npf["t2.weight"], npf["b2"], out_axis=1)
    out = quantize_layer(npf["out.weight"].reshape(3, C3), npf["b_out"], out_axis=0)
    return QModel(fc=fc, t0=t0, t1=t1, t2=t2, out=out, lut=build_tanh_lut(), c0=C0)


# Section order in the single packed payload. Renderers read slices by these
# offsets, so the order here IS the payload layout.
SECTION_ORDER = ["FC_W", "FC_S", "FC_B", "T0_W", "T0_S", "T0_B",
                 "T1_W", "T1_S", "T1_B", "T2_W", "T2_S", "T2_B",
                 "OUT_W", "OUT_S", "OUT_B", "TANH_LUT"]


def build_sections(m: QModel) -> "dict[str, bytes]":
    b = {}
    for name, L, kind in (("FC", m.fc, "linear"), ("T0", m.t0, "tconv"),
                          ("T1", m.t1, "tconv"), ("T2", m.t2, "tconv"),
                          ("OUT", m.out, "linear")):
        b[f"{name}_W"] = _w_bytes(L.wq, kind)
        b[f"{name}_S"] = _u32(L.scale)
        b[f"{name}_B"] = _i32(L.bias)
    b["TANH_LUT"] = m.lut.astype(np.uint8).tobytes()
    return b


def pack_payload(sections: "dict[str, bytes]"):
    """Concatenate sections in SECTION_ORDER; return (payload, offsets dict)."""
    payload = bytearray()
    offsets = {}
    for name in SECTION_ORDER:
        offsets[name] = {"offset": len(payload), "length": len(sections[name])}
        payload += sections[name]
    return bytes(payload), offsets


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    m = load_qmodel()
    sections = build_sections(m)
    payload, offsets = pack_payload(sections)
    open(os.path.join(OUTDIR, "payload.hex"), "w").write(payload.hex())
    manifest = {"act_shift": ACT_SHIFT, "scale_shift": SCALE_SHIFT,
                "lut_n": LUT_N, "lut_rsh": LUT_RSH, "lut_off": LUT_OFF, "c0": m.c0,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "sections": offsets}
    open(os.path.join(OUTDIR, "manifest.json"), "w").write(json.dumps(manifest, indent=2))
    print(f"payload: {len(payload)} bytes -> {OUTDIR}/payload.hex")
    for name in SECTION_ORDER:
        print(f"  {name:9} off {offsets[name]['offset']:6}  len {offsets[name]['length']:6}")


if __name__ == "__main__":
    main()
