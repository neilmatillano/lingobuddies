# Makes two French voices for Petit Français: girl (ff_siwis) and boy (ff_siwis + im_nicola blend).
# Both are then pitched up with ffmpeg rubberband to sound younger (girl +2.5, boy +3 semitones).
#
# Kokoro garbles single words said on their own: it adds a sound before them, so "un" and "deux"
# came out as "l'un" and "le deux". Single words are therefore said after a short lead-in
# ("Voici : un.") and the word is cut out at the pause. If a clip still sounds wrong, try another
# lead-in from LEADINS or a speed of 0.85/0.95. Nov 2026: this fixed un, deux, trois, bleu, rose,
# maman, dix (girl) and un, deux, trois, bleu, vert, rouge, blanc, maman, salut (boy).
import json, wave, numpy as np, re, os
from kokoro_onnx import Kokoro
k = Kokoro('kokoro.onnx', 'voices.bin')
V = {'girl': k.get_voice_style('ff_siwis'),
     'boy': 0.45 * k.get_voice_style('ff_siwis') + 0.55 * k.get_voice_style('im_nicola')}
LEADINS = ['Voici :', 'Regarde :', 'Et maintenant :', 'Écoute bien :']

def frames_db(s, hop):
    return np.array([20 * np.log10(np.sqrt(np.mean(s[i:i + hop] ** 2)) + 1e-7) for i in range(0, len(s) - hop, hop)])

def single_word(word, style, speed=0.9, leadin='Voici :'):
    """Say `word` after a lead-in and return only the word, cut at the quietest point between them."""
    c, sr = k.create(leadin, voice=style, speed=speed, lang='fr-fr')
    d = frames_db(c, int(sr * 0.01)); lead_end = len(d) - 1
    while lead_end > 0 and d[lead_end] < -45: lead_end -= 1
    s, sr = k.create(leadin + ' ' + word, voice=style, speed=speed, lang='fr-fr')
    hop = int(sr * 0.01); d = frames_db(s, hop)
    end = len(d) - 1
    while end > 0 and d[end] < -50: end -= 1
    lo, hi = lead_end - 25, min(lead_end + 25, end - 22)
    cut = lo + int(np.argmin(d[lo:hi]))
    w = s[cut * hop:(end + 4) * hop].astype(np.float32).copy()
    f = int(sr * 0.004); w[:f] *= np.linspace(0, 1, f); w[-f * 5:] *= np.linspace(1, 0, f * 5)
    return w, sr

clips = json.load(open('clips.json'))
for name, style in V.items():
    os.makedirs(f'wav2/{name}', exist_ok=True)
    for c in clips:
        t = c['text']
        if not re.search(r'[.!?]$', t): t += '.'
        if ' ' not in t.strip():
            s, sr = single_word(t, style)
        else:
            s, sr = k.create(t, voice=style, speed=0.9, lang='fr-fr')
        pad = np.zeros(int(sr * 0.08), dtype=np.float32)
        pcm = (np.clip(np.concatenate([pad, s.astype(np.float32), pad]), -1, 1) * 32767).astype('<i2')
        with wave.open(f"wav2/{name}/{c['id']}.wav", 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
print('done')
