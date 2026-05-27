"""
Mix a MusicGen beat with a Bark / XTTS vocal into a single track.

Closes the TODO in hanumankind_song.py: load beat + vocal, normalize rates
and lengths, blend with simple gains, write a mono WAV.

CPU only. No model deps. scipy + numpy only.

Usage:
    python mix.py <beat_path> <vocal_path> <output_path>
                  [--beat-gain 0.55] [--vocal-gain 0.85]
                  [--vocal-offset 0.0]

Defaults are tuned for MusicGen-small (32kHz mono) beat + Bark
(24kHz mono) vocal. Output sample rate matches the beat.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_mono(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return sr, x


def resample_to(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    from math import gcd
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(x, up, down)


def pad_or_loop(x, target_len, loop=False):
    if len(x) >= target_len:
        return x[:target_len]
    if not loop:
        return np.concatenate([x, np.zeros(target_len - len(x), dtype=x.dtype)])
    n = (target_len // len(x)) + 1
    return np.tile(x, n)[:target_len]


def mix(beat_path, vocal_path, out_path,
        beat_gain=0.55, vocal_gain=0.85, vocal_offset_s=0.0):
    beat_sr, beat = load_mono(beat_path)
    vocal_sr, vocal = load_mono(vocal_path)

    # Resample vocal to beat's rate; the beat sets the master rate.
    vocal = resample_to(vocal, vocal_sr, beat_sr)
    sr = beat_sr

    # Offset the vocal by N seconds of silence at the front.
    if vocal_offset_s > 0:
        vocal = np.concatenate([np.zeros(int(vocal_offset_s * sr), dtype=vocal.dtype), vocal])

    # Final length = the longer of (beat, vocal+offset). Beat loops if shorter.
    target_len = max(len(beat), len(vocal))
    beat = pad_or_loop(beat, target_len, loop=True)
    vocal = pad_or_loop(vocal, target_len, loop=False)

    mixed = beat_gain * beat + vocal_gain * vocal

    # Peak-normalize to -0.5 dBFS to leave a sliver of headroom.
    peak = float(np.max(np.abs(mixed)))
    if peak > 0:
        mixed = mixed * (10 ** (-0.5 / 20) / peak)

    out_int16 = np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(out_path, sr, out_int16)
    return out_path, sr, len(out_int16) / sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("beat", type=Path)
    p.add_argument("vocal", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--beat-gain", type=float, default=0.55)
    p.add_argument("--vocal-gain", type=float, default=0.85)
    p.add_argument("--vocal-offset", type=float, default=0.0,
                   help="seconds of silence prepended to vocal")
    args = p.parse_args()

    if not args.beat.exists():
        sys.exit(f"beat not found: {args.beat}")
    if not args.vocal.exists():
        sys.exit(f"vocal not found: {args.vocal}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    path, sr, dur = mix(
        args.beat, args.vocal, args.out,
        beat_gain=args.beat_gain, vocal_gain=args.vocal_gain,
        vocal_offset_s=args.vocal_offset,
    )
    print(f"wrote {path}  sr={sr}  duration={dur:.2f}s")


if __name__ == "__main__":
    main()
