"""
Remote Whisper backend: OpenAI-compatible `POST /audio/transcriptions` client.

Used when WHISPER_BACKEND=api (e.g. a LiteLLM / OpenAI-compatible gateway that
serves a Whisper model). The streaming state machine expects word-level
timestamps and probabilities (as produced by faster-whisper); this module
normalises whatever the server returns into that shape:

1. `words` present (verbose_json + timestamp_granularities[]=word)
   -> real word timestamps; punctuation is restored from the segment/full text
      when the word list matches it token-for-token.
2. `segments[].words` present (whisper.cpp server behind the gateway, e.g.
   Gensenta 172.17.1.65:4444) -> these are sub-word *tokens* (' Mer' + 'haba',
   punctuation as its own token, and a word can even continue in the next
   segment: '...kalite kont' | 'rol raporunu'). Tokens without a leading space
   are glued onto the previous word; the text is rebuilt from the merged words.
3. only `segments` present -> words are synthesised inside each segment's
   [start, end] window (real gaps between segments are preserved).
4. only `text` present -> words are synthesised over the whole audio with no
   gaps; finals are then produced by the silence flush / max-buffer guard.

This module intentionally has no torch / faster-whisper dependency so it can
be unit-tested in isolation.
"""

import math
import re
import threading
from typing import List, Optional, Tuple

import httpx

from skynet.logs import get_logger

log = get_logger(__name__)

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


class WhisperApiError(RuntimeError):
    """Raised for any transport / HTTP / parsing failure (caught by State)."""


def _get_client(timeout_s: float, max_connections: int) -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    timeout=httpx.Timeout(timeout_s, connect=5.0),
                    limits=httpx.Limits(
                        max_connections=max_connections, max_keepalive_connections=max_connections
                    ),
                )
    return _client


# Whisper's stock hallucinations on silence / noise / very short audio (subtitle
# credits from its training data). Gensenta's whisper.cpp returns them with high
# confidence and no_speech_prob=0, so they can only be filtered by text.
_HALLUCINATIONS = {
    'izlediğiniz için teşekkür ederim',
    'izlediğiniz için teşekkürler',
    'izlediğiniz için teşekkür ederiz',
    'abone olmayı unutmayın',
    'altyazı mk',
    'altyazı m k',
    'thank you for watching',
    'thanks for watching',
}


def _norm_text(text: str) -> str:
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    return ' '.join(''.join(ch if ch.isalnum() else ' ' for ch in text).split())


def is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    # "[Müzik]", "(alkış)", "*" ... -> nothing but annotations / symbols
    if not any(ch.isalnum() for ch in re.sub(r'\[[^\]]*\]|\([^)]*\)', ' ', stripped)):
        return True
    return _norm_text(stripped) in _HALLUCINATIONS


def _drop_hallucinated_segments(payload: dict) -> dict:
    segments = payload.get('segments') or []
    kept = [s for s in segments if not is_hallucination(str(s.get('text', '')))]
    if len(kept) == len(segments) and not is_hallucination(str(payload.get('text', ''))):
        return payload
    log.debug(f'whisper api: dropped hallucinated segment(s): {payload.get("text", "")!r}')
    text = '' if not kept else ''.join(str(s.get('text', '')) for s in kept)
    return {**payload, 'segments': kept, 'text': text, 'words': [] if len(kept) != len(segments) else payload.get('words')}


def _clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, value))


def _segment_probability(segment: dict, default_probability: float) -> float:
    avg_logprob = segment.get('avg_logprob')
    if isinstance(avg_logprob, (int, float)):
        return _clamp_prob(math.exp(avg_logprob))
    return default_probability


def _synthesise_words(text: str, start: float, end: float, probability: float) -> List[dict]:
    tokens = text.split()
    if not tokens:
        return []
    end = max(end, start)
    step = (end - start) / len(tokens) if end > start else 0.0
    words = []
    for i, token in enumerate(tokens):
        w_start = start + i * step
        # contiguous words (no artificial gaps) so gap-based cutting does not
        # split inside a synthesised segment
        w_end = start + (i + 1) * step if step else start
        words.append({'word': f' {token}', 'start': round(w_start, 3), 'end': round(w_end, 3), 'probability': probability})
    return words


def _restore_punctuation(words: List[dict], text: str) -> None:
    """
    OpenAI-style word lists usually strip punctuation. If the full text splits
    into exactly as many tokens as there are words, replace each word with the
    punctuated token so sentence-boundary detection keeps working.
    """
    tokens = text.split()
    if len(tokens) != len(words):
        return
    for word, token in zip(words, tokens):
        stripped_word = ''.join(ch for ch in word['word'] if ch.isalnum()).lower()
        stripped_token = ''.join(ch for ch in token if ch.isalnum()).lower()
        if stripped_word == stripped_token:
            word['word'] = f' {token}'


def snap_cut_to_quiet(
    pcm: bytes,
    target_s: float,
    *,
    sample_rate: int = 16000,
    max_shift_s: float = 0.6,
    frame_ms: int = 20,
    min_quiet_ms: int = 80,
) -> Optional[float]:
    """
    Remote Whisper servers (whisper.cpp) return word timestamps that can be off
    by a few hundred ms. Cutting the buffer exactly at such a timestamp splits a
    word: its tail is transcribed again in the next buffer (duplicated word) or
    lost (truncated word). This moves the cut to the middle of the closest
    really quiet stretch of the audio (energy based, relative to this buffer's
    own noise floor) within +-max_shift_s of the requested position.

    Returns the cut position in seconds, or None when there is no quiet stretch
    nearby -> the caller should not cut now (the phrase is finalised later by
    the next pause, the silence flush or the max-buffer guard).
    """
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32)
    frame = sample_rate * frame_ms // 1000
    n_frames = len(samples) // frame
    if n_frames < 5:
        return None
    rms = np.sqrt(np.mean(samples[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    # noise floor = quietest 3-frame stretch (a pause can be far shorter than
    # 10 % of the buffer, so a percentile would miss it); smoothing ignores
    # single-frame dropouts inside words
    smoothed = np.convolve(rms, np.ones(3) / 3, mode='valid')
    floor, loud = float(smoothed.min()), float(np.percentile(rms, 90))
    if loud <= 0 or loud - floor < 1e-6:
        return None
    quiet = rms <= floor + 0.15 * (loud - floor)

    min_frames = max(1, min_quiet_ms // frame_ms)
    total_s = n_frames * frame_ms / 1000
    # prefer the LONGEST quiet stretch in the window (a sentence pause beats a
    # short gap between two words), then the one closest to the target
    best, best_key = None, None
    i = 0
    while i < n_frames:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n_frames and quiet[j]:
            j += 1
        if j - i >= min_frames:
            mid = (i + j) / 2 * frame_ms / 1000
            dist = abs(mid - target_s)
            key = (j - i, -dist)
            # never cut at the very start / end of the buffer
            if dist <= max_shift_s and 0.2 < mid < total_s - 0.05 and (best_key is None or key > best_key):
                best, best_key = mid, key
        i = j
    return None if best is None else round(best, 3)


def _merge_segment_tokens(segments: List[dict], default_probability: float) -> List[dict]:
    """
    Merge whisper.cpp style sub-word tokens from `segments[].words` into words.
    A token that does not start with whitespace continues the previous word
    (also across segment boundaries). Word probability = mean of its tokens
    (same as faster-whisper); start/end come from the first/last token.
    """
    words: List[dict] = []
    counts: List[int] = []
    for seg in segments:
        seg_prob = _segment_probability(seg, default_probability)
        for tok in seg.get('words') or []:
            piece = str(tok.get('word', ''))
            stripped = piece.strip()
            # skip empty tokens and whisper.cpp special tokens like [_BEG_] / [_TT_150]
            if not stripped or (stripped.startswith('[_') and stripped.endswith(']')):
                continue
            start = float(tok.get('start', 0.0))
            end = max(float(tok.get('end', start)), start)
            prob = tok.get('probability')
            prob = _clamp_prob(float(prob)) if isinstance(prob, (int, float)) else seg_prob
            if words and not piece[0].isspace():
                word = words[-1]
                word['word'] += stripped
                word['end'] = max(word['end'], end)
                counts[-1] += 1
                word['probability'] += (prob - word['probability']) / counts[-1]
            else:
                words.append({'word': f' {stripped}', 'start': start, 'end': end, 'probability': prob})
                counts.append(1)
    return words


def normalise_response(
    payload: dict, audio_duration_s: float, default_probability: float
) -> Tuple[str, List[dict]]:
    """Return (text, words[]) where each word has word/start/end/probability."""
    payload = _drop_hallucinated_segments(payload)
    text = (payload.get('text') or '').strip()
    segments = payload.get('segments') or []
    raw_words = payload.get('words') or []

    if raw_words:
        words = []
        for w in raw_words:
            token = str(w.get('word', '')).strip()
            if not token:
                continue
            start = float(w.get('start', 0.0))
            end = float(w.get('end', start))
            prob = w.get('probability')
            if not isinstance(prob, (int, float)):
                prob = default_probability
                for seg in segments:
                    if seg.get('start', 0.0) <= start <= seg.get('end', 0.0):
                        prob = _segment_probability(seg, default_probability)
                        break
            words.append({'word': f' {token}', 'start': start, 'end': end, 'probability': _clamp_prob(float(prob))})
        _restore_punctuation(words, text)
        return text, words

    if any(seg.get('words') for seg in segments):
        words = _merge_segment_tokens(segments, default_probability)
        if words:
            # payload text may break words at segment boundaries ('kont\nrol')
            return ''.join(w['word'] for w in words).strip(), words

    if segments:
        words = []
        for seg in segments:
            words.extend(
                _synthesise_words(
                    str(seg.get('text', '')),
                    float(seg.get('start', 0.0)),
                    float(seg.get('end', audio_duration_s)),
                    _segment_probability(seg, default_probability),
                )
            )
        if not text:
            text = ' '.join(str(seg.get('text', '')).strip() for seg in segments).strip()
        return text, words

    return text, _synthesise_words(text, 0.0, audio_duration_s, default_probability)


def transcribe_wav(
    wav_bytes: bytes,
    audio_duration_s: float,
    *,
    url: str,
    api_key: str | None,
    model: str,
    language: str | None,
    prompt: str | None,
    word_timestamps: bool,
    timeout_s: float,
    max_connections: int,
    default_probability: float,
) -> Tuple[str, List[dict]]:
    endpoint = url.rstrip('/') + '/audio/transcriptions'
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}

    # httpx multipart: dict values may be lists (repeated form fields)
    data: dict[str, str | list[str]] = {'model': model, 'temperature': '0'}
    if language:
        data['language'] = language
    if prompt:
        data['prompt'] = prompt
    if word_timestamps:
        data['response_format'] = 'verbose_json'
        data['timestamp_granularities[]'] = ['word', 'segment']
    else:
        data['response_format'] = 'json'

    files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
    client = _get_client(timeout_s, max_connections)
    try:
        response = client.post(endpoint, headers=headers, data=data, files=files)
    except httpx.HTTPError as e:
        raise WhisperApiError(f'whisper api request failed: {type(e).__name__}: {e}') from e

    if response.status_code >= 400:
        # never log the request headers (API key); body is truncated
        raise WhisperApiError(f'whisper api HTTP {response.status_code}: {response.text[:300]}')

    try:
        payload = response.json()
    except ValueError as e:
        raise WhisperApiError(f'whisper api returned non-JSON body: {response.text[:300]}') from e

    if not isinstance(payload, dict):
        raise WhisperApiError(f'whisper api returned unexpected payload type {type(payload).__name__}')

    return normalise_response(payload, audio_duration_s, default_probability)
