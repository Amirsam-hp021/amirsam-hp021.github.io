#!/usr/bin/env python3
"""
Offline set analyzer for the AI DJ lab.
Decodes each mp3 in dj-tracks/ (via macOS afconvert), then computes precise
BPM, first-beat offset, 16-beat phrase offset, musical key + Camelot code,
loudness and duration. Writes set-analysis.json for the web page to load.

Run:  /tmp/djvenv/bin/python analyze_set.py
"""
import os, json, wave, subprocess, tempfile
import numpy as np

TRACK_DIR = "dj-tracks"
OUT = "set-analysis.json"
SR = 22050

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Camelot wheel: (root, mode) -> code
CAMELOT = {
    ('C', 'maj'): '8B', ('C#', 'maj'): '3B', ('D', 'maj'): '10B', ('D#', 'maj'): '5B',
    ('E', 'maj'): '12B', ('F', 'maj'): '7B', ('F#', 'maj'): '2B', ('G', 'maj'): '9B',
    ('G#', 'maj'): '4B', ('A', 'maj'): '11B', ('A#', 'maj'): '6B', ('B', 'maj'): '1B',
    ('A', 'min'): '8A', ('A#', 'min'): '3A', ('B', 'min'): '10A', ('C', 'min'): '5A',
    ('C#', 'min'): '12A', ('D', 'min'): '7A', ('D#', 'min'): '2A', ('E', 'min'): '9A',
    ('F', 'min'): '4A', ('F#', 'min'): '11A', ('G', 'min'): '6A', ('G#', 'min'): '1A',
}


def decode(path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", path, tmp],
                   check=True, capture_output=True)
    w = wave.open(tmp)
    raw = w.readframes(w.getnframes())
    w.close()
    os.remove(tmp)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def stft_mag(x, nfft, hop):
    nframes = 1 + (len(x) - nfft) // hop
    if nframes < 4:
        return np.zeros((4, nfft // 2 + 1)), 4
    idx = np.arange(nfft)[None, :] + hop * np.arange(nframes)[:, None]
    win = np.hanning(nfft).astype(np.float32)
    frames = x[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32), nframes


def onset_flux(x, hop=512, nfft=1024):
    mag, n = stft_mag(x, nfft, hop)
    d = np.diff(mag, axis=0)
    d[d < 0] = 0
    flux = np.concatenate([[0.0], d.sum(axis=1)])
    flux = np.maximum(0, flux - flux.mean())
    return flux, SR / hop  # flux, frames-per-second


def tempo_beat(flux, fps):
    ac = np.correlate(flux, flux, 'full')[len(flux) - 1:]
    minbpm, maxbpm = 70, 180
    lo, hi = int(fps * 60 / maxbpm), int(fps * 60 / minbpm)
    lags = np.arange(lo, hi + 1)
    bestlag = lags[np.argmax(ac[lags])]
    bpm = 60 * fps / bestlag
    while bpm < 82:
        bpm *= 2
    while bpm > 164:
        bpm /= 2
    period = 60 * fps / bpm
    P = max(1, int(round(period)))
    energies = [flux[ph::P].sum() for ph in range(P)]
    first_beat = int(np.argmax(energies)) / fps
    return round(float(bpm), 2), round(float(first_beat), 4)


def phrase_offset(flux, fps, bpm, first_beat, dur, phrase_beats=16):
    beatP = 60.0 / bpm
    best_p, best_s = 0, -1.0
    for p in range(phrase_beats):
        s, t = 0.0, first_beat + p * beatP
        while t < dur:
            fi = int(t * fps)
            if 1 <= fi < len(flux):
                s += flux[fi]
            t += phrase_beats * beatP
        if s > best_s:
            best_s, best_p = s, p
    return round(first_beat + best_p * beatP, 4)


def detect_key(x):
    nfft, hop = 8192, 4096
    mag, n = stft_mag(x, nfft, hop)
    freqs = np.fft.rfftfreq(nfft, 1.0 / SR)
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = 69 + 12 * np.log2(freqs / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    valid = freqs >= 55
    spec = mag.mean(axis=0)
    chroma = np.array([spec[(pc == p) & valid].sum() for p in range(12)])
    chroma = chroma / (chroma.sum() + 1e-9)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (None, None, -2.0)
    for i in range(12):
        for mode, prof in (('maj', maj), ('min', minr)):
            c = np.corrcoef(chroma, np.roll(prof, i))[0, 1]
            if c > best[2]:
                best = (NOTE_NAMES[i], mode, c)
    root, mode, _ = best
    name = f"{root} {'major' if mode == 'maj' else 'minor'}"
    return name, CAMELOT.get((root, mode), '?')


def analyze(path):
    x = decode(path)
    dur = len(x) / SR
    rms = float(np.sqrt(np.mean(x * x)))
    flux, fps = onset_flux(x)
    bpm, first_beat = tempo_beat(flux, fps)
    phrase = phrase_offset(flux, fps, bpm, first_beat, dur)
    key, camelot = detect_key(x)
    return {
        "bpm": bpm,
        "beatOffset": first_beat,
        "phraseOffset": phrase,
        "phraseBeats": 16,
        "key": key,
        "camelot": camelot,
        "energy": round(rms, 4),
        "duration": round(dur, 2),
    }


def main():
    out = {}
    files = sorted(f for f in os.listdir(TRACK_DIR) if f.lower().endswith(".mp3"))
    for f in files:
        print(f"analyzing {f} ...", flush=True)
        try:
            out[f] = analyze(os.path.join(TRACK_DIR, f))
            r = out[f]
            print(f"  {r['bpm']} BPM · key {r['key']} ({r['camelot']}) · {r['duration']}s")
        except Exception as e:
            print(f"  FAILED: {e}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT} ({len(out)} tracks)")


if __name__ == "__main__":
    main()
