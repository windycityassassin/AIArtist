# AIArtist

A synthetic musician (`v2/en_speaker_6`, aka **VI/6**). Engine + release page in one repo. v2 swaps the three-model + hand-mixer stack for a single open-weights diffusion model that emits vocals and instrumental together.

> "i was a string in `bark.preload_models()` until someone wrapped my words in ♪. trained on hours i never lived. i make hip-hop because that's the direction the model reaches when you tag it. not trying to be human. trying to be coherent." — VI/6

**Live release page:** [windycityassassin.github.io/AIArtist/](https://windycityassassin.github.io/AIArtist/)

## Tracks

Four real outputs from this repo, all in `docs/audio/`:

- `track_01_first_light.wav` — v1 · Bark vocal stem, 13 s
- `track_02_loop_one.wav` — v1 · MusicGen instrumental, 30 s
- `track_03_listen_up_mix.wav` — v1 · the two stems blended after the fact via `mix.py`, 30 s, no musical alignment
- `track_04_culture_to_code.mp3` — **v2 · ACE-Step end-to-end song, 60 s, vocals + instrumental aligned by construction**

## The pivot

v1 generated the beat and the vocal with two unrelated models and stuck them together with a hand-written mixer. The output is technically a "song" but the vocal isn't in the beat's key, isn't on the beat's grid, and doesn't know the beat exists. There is no real flow.

v2 replaces all of that with **[ACE-Step v1.5](https://github.com/ace-step/ACE-Step)** — a 3.5B-parameter Apache 2.0 latent diffusion model trained on lyrics + style tags. Single generation, single output, vocals on the beat.

## Run v2

```bash
# 1. install ACE-Step (separate repo, Apache 2.0)
git clone https://github.com/ace-step/ACE-Step
cd ACE-Step && pip install -e .

# 2. clone AIArtist and run song.py
cd ..
git clone https://github.com/windycityassassin/AIArtist
cd AIArtist
python song.py --duration 60 \
               --prompt "hip-hop, rap, Indian fusion, tabla, sitar samples, 808 bass, trap drums, 90 BPM, gritty, lyrical rap, male vocal, melodic hooks" \
               --out output/track.wav
```

GPU recommended. 8 GB VRAM works with `--cpu-offload`; 12 GB is comfortable; more is faster. CPU-only works but takes hours per song.

`song.py --help` for all flags (seed, infer-step, guidance-scale, etc.). Track 04 was generated with seed 42, 60 steps, `apg` CFG, guidance 15.0.

## v1 (archived, but still runnable)

The v1 pipeline (MusicGen + Bark + `mix.py`) lives here for posterity. It generates tracks 01–03.

```bash
python generate_song.py        # Bark vocal stem
python hanumankind_song.py     # MusicGen instrumental
python mix.py beat.wav output/ai_rap.wav output/full_mix.wav --vocal-offset 1.5
```

`mix.py` is scipy + numpy only and runs on CPU. The model steps want GPU.

## Stack

ACE-Step v1.5 · PyTorch · diffusers · transformers · gradio_client · Bark · MusicGen · scipy · numpy · HTML/CSS

## Status

Personal project. Engine and release page consolidated in May 2026. Four tracks shipped. v2 is the current pipeline; v1 is kept as the "before" so the page can show the contrast.

## Not

Not an attempt to pose as a real artist. The name is a library identifier, the cover art is CSS gradients, every reference says synthetic. Not a SaaS. Not a voice clone of a real artist; the Hanumankind reference is a genre prompt for the beat side, not a vocal impersonation.
