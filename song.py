"""
AIArtist v2 — lyric-to-song generation via ACE-Step.

Replaces the v1 stack (MusicGen-small + Bark + mix.py) with a single
model that emits aligned vocals and instrumental together. Same lyric +
style prompt interface; very different output quality.

Requires the ACE-Step pipeline installed:

    git clone https://github.com/ace-step/ACE-Step
    cd ACE-Step && pip install -e .

GPU is strongly recommended. 8 GB VRAM minimum with --cpu-offload, 12 GB
comfortable, more is faster. CPU-only inference is technically possible
but takes hours per song.

Usage:

    python song.py --duration 60 \\
                   --prompt "hip-hop, Indian fusion, 90 BPM, gritty" \\
                   --lyrics-file lyrics.txt \\
                   --out output/track.wav

Lyrics use ACE-Step section markers, e.g.:

    [verse]
    Living life like a movie scene
    Breaking barriers, living my dream
    ...

    [chorus]
    Mix the culture with the flow
    ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_PROMPT = (
    "hip-hop, rap, Indian fusion, tabla, sitar samples, 808 bass, "
    "trap drums, 90 BPM, gritty, lyrical rap, male vocal, melodic hooks"
)

DEFAULT_LYRICS = """[verse]
Living life like a movie scene
Breaking barriers, living my dream
From Bangalore streets to worldwide beats
Every verse I drop is pure heat

[verse]
Mix the culture with the flow
Ancient wisdom with the modern show
Breaking boundaries as I go
This is how we make it grow
"""


def load_lyrics(path: Path | None) -> str:
    if path is None:
        return DEFAULT_LYRICS
    return path.read_text(encoding="utf-8")


def build_pipeline(checkpoint_dir: str | None, dtype: str, cpu_offload: bool):
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError:
        sys.exit(
            "acestep is not installed. Install it with:\n"
            "    git clone https://github.com/ace-step/ACE-Step\n"
            "    cd ACE-Step && pip install -e ."
        )
    return ACEStepPipeline(
        checkpoint_dir=checkpoint_dir or "",
        dtype=dtype,
        torch_compile=False,
        cpu_offload=cpu_offload,
        overlapped_decode=cpu_offload,
    )


def generate(
    pipeline,
    duration: float,
    prompt: str,
    lyrics: str,
    out_path: Path,
    seed: int = 42,
    infer_step: int = 60,
    guidance_scale: float = 15.0,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline(
        audio_duration=duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=str(seed),
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        use_erg_tag=True,
        use_erg_lyric=False,
        use_erg_diffusion=True,
        oss_steps="",
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        save_path=str(out_path),
    )
    return out_path


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0] if __doc__ else None)
    p.add_argument("--duration", type=float, default=60.0,
                   help="seconds of audio to generate (default 60)")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                   help="style tags: genre, instruments, BPM, mood")
    p.add_argument("--lyrics-file", type=Path, default=None,
                   help="path to lyrics file (default: built-in example)")
    p.add_argument("--out", type=Path, default=Path("output/track.wav"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer-step", type=int, default=60)
    p.add_argument("--guidance-scale", type=float, default=15.0)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to local ACE-Step checkpoint dir (auto-downloads from HF if omitted)")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--cpu-offload", action="store_true",
                   help="enable for low-VRAM GPUs (<12 GB)")
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    lyrics = load_lyrics(args.lyrics_file)
    pipeline = build_pipeline(args.checkpoint, args.dtype, args.cpu_offload)
    out = generate(
        pipeline,
        duration=args.duration,
        prompt=args.prompt,
        lyrics=lyrics,
        out_path=args.out,
        seed=args.seed,
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
