"""
Probe an OpenAI-compatible Whisper endpoint before switching skynet to
WHISPER_BACKEND=api. Reports whether word timestamps / segments / probabilities
are returned and how long a request takes.

Usage (no third-party deps besides httpx):
    WHISPER_API_KEY=... python tools/probe_whisper_api.py \
        --url http://<host>:<port>/v1 --model whisper --lang tr sample.wav [--runs 5]
"""

import argparse
import json
import os
import statistics
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--url', default=os.environ.get('WHISPER_API_URL', ''))
    ap.add_argument('--model', default=os.environ.get('WHISPER_API_MODEL', 'whisper'))
    ap.add_argument('--lang', default='tr')
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()

    key = os.environ.get('WHISPER_API_KEY')
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    base = args.url.rstrip('/')
    audio = open(args.audio, 'rb').read()

    with httpx.Client(timeout=60) as client:
        try:
            r = client.get(f'{base}/models', headers=headers)
            print(f'GET /models -> {r.status_code} {r.text[:300]}')
        except httpx.HTTPError as e:
            print(f'GET /models failed: {e}')

        variants = {
            'verbose_json+word': {'response_format': 'verbose_json', 'timestamp_granularities[]': ['word', 'segment']},
            'verbose_json': {'response_format': 'verbose_json'},
            'json': {'response_format': 'json'},
        }
        for name, extra in variants.items():
            data = {'model': args.model, 'language': args.lang, 'temperature': '0', **extra}
            durations, payload, status = [], None, None
            for _ in range(args.runs):
                t0 = time.perf_counter()
                r = client.post(f'{base}/audio/transcriptions', headers=headers, data=data,
                                files={'file': ('audio.wav', audio, 'audio/wav')})
                durations.append(time.perf_counter() - t0)
                status = r.status_code
                if r.status_code >= 400:
                    print(f'[{name}] HTTP {r.status_code}: {r.text[:300]}')
                    break
                payload = r.json()
            if payload is None:
                continue
            words = payload.get('words') or []
            segs = payload.get('segments') or []
            print(f'\n[{name}] HTTP {status}  latency avg {statistics.mean(durations):.2f}s '
                  f'min {min(durations):.2f}s max {max(durations):.2f}s')
            print(f'  keys: {sorted(payload.keys())}')
            print(f'  text: {payload.get("text", "")[:200]}')
            print(f'  words: {len(words)}  sample: {json.dumps(words[:3], ensure_ascii=False)}')
            print(f'  word probability: {"yes" if words and "probability" in words[0] else "no"}')
            print(f'  segments: {len(segs)}  avg_logprob: {"yes" if segs and "avg_logprob" in segs[0] else "no"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
