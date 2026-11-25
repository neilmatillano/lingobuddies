# Makes two French voices for Petit Français: girl (ff_siwis) and boy (ff_siwis + im_nicola blend).
# Both are then pitched up with ffmpeg rubberband to sound younger (see encode step).
import json, wave, numpy as np, re, os
from kokoro_onnx import Kokoro
k = Kokoro('kokoro.onnx', 'voices.bin')
V = {'girl': k.get_voice_style('ff_siwis'),
     'boy': 0.45 * k.get_voice_style('ff_siwis') + 0.55 * k.get_voice_style('im_nicola')}
clips = json.load(open('clips.json'))
for name, style in V.items():
    os.makedirs(f'wav2/{name}', exist_ok=True)
    for c in clips:
        t = c['text']
        if not re.search(r'[.!?]$', t): t += '.'
        s, sr = k.create(t, voice=style, speed=0.9, lang='fr-fr')
        pad = np.zeros(int(sr * 0.08), dtype=np.float32)
        pcm = (np.clip(np.concatenate([pad, s.astype(np.float32), pad]), -1, 1) * 32767).astype('<i2')
        with wave.open(f"wav2/{name}/{c['id']}.wav", 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
print('done')
