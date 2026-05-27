# AIArtist

A synthetic musician named after a Bark voice preset (`v2/en_speaker_6`). Engine + release page in one repo. MusicGen-small writes the beats, Bark (or XTTS v2) carries the vocals, `mix.py` glues them together.

**Live release page:** [windycityassassin.github.io/AIArtist/](https://windycityassassin.github.io/AIArtist/)

## Tracks

Three real outputs from this engine ship in `docs/audio/`:

- `track_01_first_light.wav` — Bark vocal stem, ~13 s
- `track_02_loop_one.wav` — MusicGen instrumental, ~30 s
- `track_03_listen_up_mix.wav` — track 01 vocal layered over track 02 beat via `mix.py`, ~30 s, the first fully-mixed render from this pipeline

## Run the pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate the vocal (Bark, 24 kHz mono)
python generate_song.py
#   -> output/ai_rap.wav

# 2. generate the beat (MusicGen, 32 kHz mono)
python hanumankind_song.py
#   -> beat.wav

# 3. mix (CPU only, no model deps)
python mix.py beat.wav output/ai_rap.wav output/full_mix.wav --vocal-offset 1.5
```

Steps 1 and 2 download ~3 GB of model weights on first run and want a GPU (CPU works but is several minutes per 10 s of audio). Step 3 runs anywhere.

## mix.py

`mix.py` closes the `TODO: Mix the beat and vocals` that previously lived at `hanumankind_song.py:58`. Resamples the vocal to the beat's rate, offsets it by a configurable lead-in, loops the beat to the vocal length, blends with simple gains, peak-normalizes to −0.5 dBFS, writes mono int16 WAV. scipy + numpy only.

```bash
python mix.py <beat> <vocal> <out> [--beat-gain 0.55] [--vocal-gain 0.85] [--vocal-offset 0.0]
```

## Stack

PyTorch · transformers (MusicGen) · TTS (XTTS v2) · bark · scipy · numpy · HTML/CSS (release page)

## Status

Personal project. Engine and release page consolidated under one repo in May 2026. Three tracks shipped. Voice-clone path (XTTS v2 with a user-supplied reference WAV) is wired in `hanumankind_song.py` and ready for the next track.

## Not

Not an attempt to pose as a real artist. The name is a library identifier, the cover art is CSS gradients, every reference says synthetic. Not a SaaS — repo is the engine and the page is the showcase, no API key in the request path. Not a voice clone of a real artist; the Hanumankind reference is a genre prompt for the beat, not a vocal impersonation.
