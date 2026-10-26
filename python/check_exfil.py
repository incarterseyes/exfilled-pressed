"""Verify the GGUF template and copy model against Pressed's independent fixtures."""
import argparse
import random
import re

from jinja2 import Environment

from exfil import (ROOT, SOURCE_COUNT, OUTPUT_CHARS, URI_PREFIX, flower_template,
                   prompt, render, request_json, decode_output, read_png)
from dump_expected import SEEDS
from export import load_qmodel
from check_numeric import model_bounds
from engine import render_rgb
from latent import seed_to_latent, hex_to_be32


def template_image(template, content):
    formatted = template.render(messages=[{'role': 'user', 'content': content}])
    tokens = [int(v) for v in re.findall(r'<p(\d{3})>', formatted)]
    assert len(tokens) == SOURCE_COUNT and tokens[-1] == 256
    text = bytes(tokens[:-1]).decode('ascii')
    assert len(text) == OUTPUT_CHARS and text.startswith(URI_PREFIX)
    return decode_output(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Also test an unchanged local llama-server')
    args = parser.parse_args()
    source = flower_template()
    environment = Environment()
    template = environment.from_string(source)
    # Exercise signed scaling at large magnitudes, where naive double products
    # lose integer bits. The same macro is embedded in the GGUF.
    macro = template.module.scaled
    model = load_qmodel()
    for (_, acc, _, _), layer in zip(model_bounds(model),
                                    [model.fc, model.t0, model.t1, model.t2, model.out]):
        scale = int(max(layer.scale))
        assert max(acc, (acc // 65536) * scale, 65535 * (65536 + scale)) < 2**53
    rng = random.Random(271828)
    pairs = [(a, s) for a in [0, 1, -1, 65535, -65535, 65536, -65536, 242762941971, -242762941971]
             for s in [1, 65535, 65536, 26300000]]
    pairs += [(rng.randrange(-242762941971, 242762941972), rng.randrange(1, 26300001)) for _ in range(1000)]
    for acc, scale in pairs:
        assert int(macro(acc, scale, -17, False)) == ((acc * scale) >> 32) - 17
    print('PASS: signed split-product scaling boundaries', flush=True)
    for i, seed in enumerate(SEEDS):
        expected = bytes.fromhex((ROOT / f'test/fixtures/expected_{i}.hex').read_text().strip()[2:])
        image = template_image(template, prompt(seed))
        rgb, latent = read_png(image)
        assert rgb == expected, f'Python template fixture {i}'
        assert latent == seed_to_latent(hex_to_be32(seed))
        if args.url:
            rgb, _ = render(args.url, seed)
            assert rgb == expected, f'llama-server fixture {i}'
        print(f'PASS: fixture {i} ({seed})', flush=True)
    random_images = []
    for timestamp in ['20260919123000', '20260919123001']:
        environment.globals['strftime_now'] = lambda _, value=timestamp: value
        image = template_image(template, 'flower')
        assert image == template_image(template, 'flower'), 'Same-second repeatability'
        rgb, latent = read_png(image)
        assert len(latent) == 32 and all(-128 <= v <= 127 for v in latent)
        assert rgb == render_rgb(model, latent), 'Random image reference parity'
        random_images.append(image)
    assert random_images[0] != random_images[1], 'Clock-seeded inputs must vary'
    corrupt = bytearray(random_images[0])
    corrupt[50] ^= 1
    try:
        read_png(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt PNG accepted')
    print('PASS: random PNGs vary, reproduce exactly, and reject corruption', flush=True)
    if args.url:
        # Every printable ASCII byte and every address, independent of GAN arithmetic.
        values = [32 + i % 95 for i in range(OUTPUT_CHARS)]
        result = request_json(args.url.rstrip('/') + '/completion', {
            'prompt': ''.join(f'<p{v:03}>' for v in values) + '<p256>',
            'n_predict': SOURCE_COUNT, 'temperature': 0})
        assert result['content'] == bytes(values).decode('ascii'), 'Copy model ASCII/address coverage'
        previous = None
        for _ in range(2):
            rgb, result = render(args.url)
            image = decode_output(result['choices'][0]['message']['content'])
            _, latent = read_png(image)
            assert rgb == render_rgb(model, latent), 'Native random image parity'
            assert image != previous, 'Sequential native random calls repeated'
            previous = image
        print('PASS: ASCII transport and two native random flower calls', flush=True)
    print('PASS: Exfil flower experiment', flush=True)


if __name__ == '__main__':
    main()
