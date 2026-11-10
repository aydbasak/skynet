"""Offline simulation test for WHISPER_BACKEND=api (torch/silero/faster-whisper stubbed).

Run: python tools/test_whisper_api_backend.py <repo-root>   (needs httpx, pydantic, numpy, uuid6, fastapi, prometheus deps)
"""
import asyncio, json, os, sys, threading, types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = {'v': 'words', 'calls': [], 'fail': False}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        fields = [f for f in [b'verbose_json', b'timestamp_granularities', b'name="language"', b'name="prompt"'] if f in body]
        MODE['calls'].append({'path': self.path, 'auth': self.headers.get('Authorization'),
                              'wav': b'RIFF' in body and b'WAVE' in body, 'fields': fields})
        if MODE['fail']:
            self.send_response(429); self.end_headers(); self.wfile.write(b'rate limited'); return
        m = MODE['v']
        if m == 'words':
            s1 = 'Merhaba arkadaşlar, bugünkü tedarikçi denetim toplantısına hoş geldiniz.'
            s2 = 'Bugün denetim bulgularını konuşacağız.'
            words, t = [], 0.0
            for sent, gap in ((s1, 0.8), (s2, 0.0)):
                for tok in sent.split():
                    words.append({'word': tok.strip('.,'), 'start': round(t, 2), 'end': round(t + 0.4, 2)})
                    t += 0.45
                t += gap
            r = {'text': s1 + ' ' + s2,
                 'segments': [{'start': 0.0, 'end': words[7]['end'], 'text': s1, 'avg_logprob': -0.15},
                              {'start': words[8]['start'], 'end': words[-1]['end'], 'text': s2, 'avg_logprob': -0.2}],
                 'words': words}
            # like a real server: audio that ends inside the pause (<= 4.2 s) only contains s1
            if len(body) / 32000 <= 4.2:
                r = {'text': s1, 'segments': r['segments'][:1], 'words': words[:8]}
        elif m == 'segwords':
            # shape observed from Gensenta's gateway (whisper.cpp): top-level words empty,
            # sub-word tokens inside segments, a word split across segments ('kont'|'rol')
            seg_tokens = [
                ['[_BEG_]', ' Mer', 'haba', ',', ' bu', ' bir', ' trans', 'kript', ' test', 'idir', '.',
                 ' Topl', 'antıda', ' kal', 'ite', ' kont'],
                ['rol', ' rap', 'or', 'unu', ' konuştuk', '.'],
                [' Saat', ' 14', "'te", ' tekrar', ' görüş', 'eceğiz', '.'],
            ]
            segs, t = [], 0.0
            for toks in seg_tokens:
                ws, s0 = [], t
                for tok in toks:
                    ws.append({'word': tok, 'start': round(t, 2), 'end': round(t + 0.2, 2), 't_dtw': -1, 'probability': 0.9})
                    t += 0.2
                segs.append({'start': round(s0, 2), 'end': round(t, 2), 'text': ''.join(toks[1:] if toks[0].startswith('[_') else toks),
                             'avg_logprob': -0.07, 'words': ws})
                t += 0.6  # pause between segments
            segs[1]['words'][0]['probability'] = 0.5
            r = {'text': " Merhaba, bu bir transkript testidir. Toplantıda kalite kont\nrol raporunu konuştuk.\n Saat 14'te tekrar görüşeceğiz.",
                 'language': 'turkish', 'segments': segs, 'words': []}
        elif m == 'segments':
            r = {'text': 'Merhaba arkadaşlar. Bugün denetim var.',
                 'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Merhaba arkadaşlar.'},
                              {'start': 3.0, 'end': 5.0, 'text': 'Bugün denetim var.'}]}
        else:
            r = {'text': 'Merhaba arkadaşlar bugün denetim var'}
        out = json.dumps(r).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(out)


srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

os.environ.update(WHISPER_BACKEND='api', WHISPER_API_URL=f'http://127.0.0.1:{srv.server_port}/v1',
                  WHISPER_API_KEY='test-key', WHISPER_API_MIN_INTERVAL_MS='0', WHISPER_API_MAX_BUFFER_S='20',
                  ENABLED_MODULES='streaming_whisper', BYPASS_AUTHORIZATION='1')

# ---- stub heavy deps (torch / silero / faster_whisper) ----
VAD = {'speech': []}
for name in ['torch', 'torchaudio']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['torch'].cuda = types.SimpleNamespace(is_available=lambda: False)
sv = types.ModuleType('silero_vad')
sv.load_silero_vad = lambda onnx=False: object()
sv.read_audio = lambda stream: stream
sv.get_speech_timestamps = lambda audio, **kw: list(VAD['speech'])
sys.modules['silero_vad'] = sv
fw = types.ModuleType('faster_whisper')
fwt = types.ModuleType('faster_whisper.tokenizer')


def _no_local(*a, **k):
    raise AssertionError('local model must not load in api mode')


fw.WhisperModel = _no_local
class _Tok:
    def __init__(self, *a, **k):
        raise AssertionError('tokenizer must not load in api mode')


fwt.Tokenizer = _Tok
sys.modules['faster_whisper'] = fw
sys.modules['faster_whisper.tokenizer'] = fwt

sys.path.insert(0, sys.argv[1])
from skynet.modules.stt.streaming_whisper.utils import utils  # noqa: E402
from skynet.modules.stt.streaming_whisper import cfg  # noqa: E402
from skynet.modules.stt.streaming_whisper.meeting_connection import MeetingConnection  # noqa: E402


def ok(c, m):
    print(('PASS ' if c else 'FAIL ') + m)
    if not c:
        sys.exit(1)


ok(cfg.use_api and cfg.model is None, 'api mode, no local model loaded')

# 1) words mode
MODE['v'] = 'words'
audio = b'\x00\x01' * 16000 * 6  # 6 s
r = utils.transcribe([audio], 'tr', 'önceki cümle')
ok(len(r.words) == 12, 'words mode: 12 words')
ok(r.words[7].word == ' geldiniz.' and r.words[1].word == ' arkadaşlar,', 'punctuation restored from text')
ok(0.8 < r.words[0].probability < 0.9, f'probability from avg_logprob ({r.words[0].probability:.3f})')
c = MODE['calls'][-1]
ok(c['path'] == '/v1/audio/transcriptions' and c['auth'] == 'Bearer test-key' and c['wav'], 'request path/auth/wav')
ok(len(c['fields']) == 4, f"verbose_json+granularity+language+prompt sent {c['fields']}")
cut = utils.get_cut_mark_from_segment_probability(r)
ok(cut.start == r.words[7].end and cut.end == r.words[8].start, f'sentence cut at punctuation gap ({cut.start}-{cut.end})')

# 1b) whisper.cpp sub-word tokens inside segments (Gensenta gateway)
MODE['v'] = 'segwords'
r = utils.transcribe([audio], 'tr', None)
texts = [w.word for w in r.words]
expected = [' Merhaba,', ' bu', ' bir', ' transkript', ' testidir.', ' Toplantıda', ' kalite', ' kontrol', ' raporunu',
            ' konuştuk.', ' Saat', " 14'te", ' tekrar', ' görüşeceğiz.']
ok(texts == expected, f'segwords: sub-word tokens merged, [_BEG_] skipped, cross-segment word joined {texts}')
kontrol = r.words[7]
ok((kontrol.start, kontrol.end) == (3.0, 4.0), f'segwords: merged word spans first..last token ({kontrol.start}-{kontrol.end})')
ok(abs(kontrol.probability - 0.7) < 1e-9 and abs(r.words[0].probability - 0.9) < 1e-9,
   f'segwords: probability = mean of token probs ({kontrol.probability:.3f})')
cut = utils.get_cut_mark_from_segment_probability(r)
ok((cut.start, cut.end) == (r.words[9].end, r.words[10].start) == (5.0, 5.6),
   f'segwords: sentence cut after "konuştuk." ({cut.start}-{cut.end})')

# 1c) cut snapping to real silence (imprecise remote word timestamps)
from skynet.modules.stt.streaming_whisper.utils import api_client  # noqa: E402
import struct  # noqa: E402


def pcm(spec):  # [(seconds, amplitude)] -> s16le square-ish signal
    out = b''
    for secs, amp in spec:
        n = int(16000 * secs)
        out += b''.join(struct.pack('<h', amp if (k // 20) % 2 else -amp) for k in range(n))
    return out


speech_gap = pcm([(2.0, 4000), (0.3, 0), (2.0, 4000)])  # real pause at 2.0-2.3 s
ok(api_client.snap_cut_to_quiet(speech_gap, 2.4) == 2.15, 'snap: cut moved into the real pause (2.4 -> 2.15)')
ok(api_client.snap_cut_to_quiet(speech_gap, 3.5) is None, 'snap: no pause within 0.6 s -> do not cut')
ok(api_client.snap_cut_to_quiet(pcm([(4.0, 4000)]), 2.0) is None, 'snap: continuous speech -> do not cut')
ok(api_client.snap_cut_to_quiet(pcm([(2.0, 4000), (0.3, 40), (2.0, 4000)]), 2.0) == 2.15,
   'snap: background noise in the pause still counts as quiet')

# 1d) stock hallucinations (Gensenta returns them for silence with no_speech_prob=0)
hal = {'text': ' İzlediğiniz için teşekkür ederim.\n',
       'segments': [{'start': 0.0, 'end': 1.0, 'text': ' İzlediğiniz için teşekkür ederim.', 'avg_logprob': -0.06,
                     'no_speech_prob': 0.0, 'words': [{'word': ' İzlediğiniz', 'start': 0.0, 'end': 0.5},
                                                      {'word': ' için', 'start': 0.5, 'end': 1.0}]}], 'words': []}
ok(api_client.normalise_response(hal, 1.0, 0.8) == ('', []), 'hallucination: silence phrase dropped')
ok(api_client.is_hallucination('[Müzik]') and api_client.is_hallucination(' (alkış) '), 'hallucination: annotation-only dropped')
ok(not api_client.is_hallucination('Toplantıya katıldığınız için teşekkür ederim.'), 'hallucination: real thanks kept')
mixed = {'text': ' Günaydın. İzlediğiniz için teşekkür ederim.',
         'segments': [{'start': 0, 'end': 1, 'text': ' Günaydın.', 'words': [{'word': ' Günaydın.', 'start': 0, 'end': 0.9}]},
                      hal['segments'][0]], 'words': []}
ok(api_client.normalise_response(mixed, 2.0, 0.8)[0] == 'Günaydın.', 'hallucination: only the bad segment is removed')

# 1e) API mode never sends empty / very short buffers
from skynet.modules.stt.streaming_whisper.state import State  # noqa: E402
from skynet.modules.stt.streaming_whisper.chunk import Chunk  # noqa: E402

n_calls = len(MODE['calls'])
st = State('p', 'tr')
silent = 'p|tr'.encode().ljust(60, b'\x00') + b'\x00\x00' * 4096
VAD['speech'] = []
asyncio.run(st.process(Chunk(silent, utils.now()), None))
ok(len(MODE['calls']) == n_calls and asyncio.run(st.force_transcription(None)) is None and len(MODE['calls']) == n_calls,
   'no request for an empty buffer (process + flush)')

# 2) segments only
MODE['v'] = 'segments'
r = utils.transcribe([audio], 'tr', None)
ok(len(r.words) == 5 and r.words[2].start == 3.0 and r.words[1].end == 2.0, 'segments mode: words synthesised inside segments')
# 3) text only
MODE['v'] = 'text'
r = utils.transcribe([audio], 'tr', None)
ok(len(r.words) == 5 and r.words[-1].end == 6.0, 'text mode: words spread over audio')
ok(utils.get_cut_mark_from_segment_probability(r).end == 0.0, 'text mode: no premature cut')


async def flow():
    MODE['v'] = 'words'
    mc = MeetingConnection(ws=None)

    n_chunk = {'i': 0}

    def chunk():
        # 0.256 s of "speech", silent where the mocked API puts the sentence
        # pause (3.55-4.4 s) so the cut can snap to real silence
        header = 'participant-1|tr'.encode().ljust(60, b'\x00')
        i = n_chunk['i']
        n_chunk['i'] += 1
        mid = (i + 0.5) * 0.256
        return header + (b'\x00\x00' * 4096 if 3.6 <= mid <= 4.3 else pcm([(0.256, 3000)]))

    out = []
    t = utils.now()
    for i in range(24):  # ~6.1 s of speech
        VAD['speech'] = [{'start': 0.0, 'end': 0.256 * (i + 1)}]
        res = await mc.process(chunk(), t + i * 256)
        if res:
            out.extend(res)
    finals = [p for p in out if p.type == 'final']
    interims = [p for p in out if p.type == 'interim']
    ok(len(finals) >= 1 and finals[0].text == 'Merhaba arkadaşlar, bugünkü tedarikçi denetim toplantısına hoş geldiniz.',
       f'final emitted: {finals[0].text if finals else None}')
    ok(len(interims) >= 1, f'interims emitted ({len(interims)})')
    ok(isinstance(mc.previous_transcription_tokens, str) and 'geldiniz' in mc.previous_transcription_tokens,
       'prompt context kept as text')
    res = await mc.force_transcription('participant-1')
    ok(bool(res) and res[0].type == 'final', 'flush produces final')
    st = mc.participants['participant-1']
    ok(st.working_audio == b'', 'buffer reset after flush')
    # API failure
    st.working_audio = b'\x01\x00' * 16000 * 2
    MODE['fail'] = True
    res = await mc.force_transcription('participant-1')
    ok(res is None and len(st.working_audio) > 0 and not st.is_transcribing, 'API 429 -> no crash, buffer kept')
    VAD['speech'] = [{'start': 0.0, 'end': 9.0}]
    res = await mc.process(chunk(), utils.now())
    ok(res is None and not st.is_transcribing, 'process during outage -> None, no crash')
    MODE['fail'] = False
    res = await mc.force_transcription('participant-1')
    ok(bool(res) and res[0].type == 'final', 'recovers after outage')
    # max buffer guard
    MODE['v'] = 'text'
    st.reset()
    st.working_audio = b'\x01\x00' * 16000 * 21
    VAD['speech'] = [{'start': 0.0, 'end': 21.3}]
    res = await mc.process(chunk(), utils.now())
    ok(bool(res) and res[0].type == 'final' and st.working_audio == b'', 'max buffer (20s) forces final')


asyncio.run(flow())
print('API calls:', len(MODE['calls']))
print('ALL TESTS PASSED')
