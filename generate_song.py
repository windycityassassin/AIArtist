import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from scipy.io.wavfile import write as write_wav
import numpy as np

def generate_ai_song(lyrics, voice_preset="v2/en_speaker_6", output_path="ai_song.wav"):
    """
    Generate an AI song with specified lyrics and voice preset
    
    Args:
        lyrics (str): The lyrics/text to be converted to song
        voice_preset (str): The voice preset to use
        output_path (str): Path to save the generated audio
    """
    # Download and load all models
    preload_models()
    
    # Generate audio from text
    print("Generating audio...")
    audio_array = generate_audio(lyrics, history_prompt=voice_preset)
    
    # Save audio to disk
    print(f"Saving audio to {output_path}")
    write_wav(output_path, SAMPLE_RATE, audio_array)
    
    return output_path

if __name__ == "__main__":
    # Example lyrics in rap style
    lyrics = """
    ♪ Listen up, this is my story to tell
    Been through the highs, been through the lows as well
    Every day I'm grinding, trying to make it right
    Got my eyes on the prize, keeping my dreams in sight ♪

    ♪ They said I couldn't make it, but I proved them wrong
    Standing here right now, singing my victory song
    This is just the beginning, watch me as I rise
    Breaking through the ceiling, touching starlit skies ♪
    """
    
    # Available voice presets:
    # v2/en_speaker_0 through v2/en_speaker_9 (different voice characteristics)
    # v2/en_speaker_6 is particularly good for singing
    
    # Generate the song with a specific voice preset
    output_file = generate_ai_song(
        lyrics,
        voice_preset="v2/en_speaker_6",  # You can change this to try different voices
        output_path="output/ai_rap.wav"
    )
    print(f"Song generated and saved to {output_file}")
