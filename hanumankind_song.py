import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
import os
from TTS.api import TTS

class AIRapGenerator:
    def __init__(self):
        # Initialize the MusicGen model for the beat
        self.music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        
        # Initialize TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
    def generate_beat(self, prompt, duration=30, output_path="beat.wav"):
        """Generate the instrumental beat"""
        inputs = self.music_processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        
        audio_values = self.music_model.generate(
            **inputs,
            max_new_tokens=duration * 50,
            do_sample=True,
            guidance_scale=3.0
        )
        
        sampling_rate = self.music_model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].numpy()
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        wavfile.write(output_path, sampling_rate, (audio_data * 32767).astype(np.int16))
        return output_path
    
    def generate_rap(self, lyrics, output_path="rap_song.wav"):
        """Generate the complete rap song with lyrics"""
        # First generate the beat
        beat_prompt = """
        Create a hip-hop beat with Indian fusion elements,
        strong bass, dynamic percussion, and melodic elements
        similar to Hanumankind's style
        """
        beat_path = self.generate_beat(beat_prompt)
        
        # Generate vocals with TTS
        vocals_path = "output/vocals.wav"
        self.tts.tts_to_file(
            text=lyrics,
            file_path=vocals_path,
            speaker_wav="output/reference.wav",
            language="en"
        )
        
        # TODO: Mix the beat and vocals
        # For now, we'll just return the vocals
        return vocals_path

def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Example rap lyrics in Hanumankind's style
    lyrics = """
    Living life like a movie scene,
    Breaking barriers, living my dream,
    From Bangalore streets to worldwide beats,
    Every verse I drop is pure heat.
    
    Mix the culture with the flow,
    Ancient wisdom with the modern show,
    Breaking boundaries as I go,
    This is how we make it grow.
    """
    
    # Initialize the generator
    generator = AIRapGenerator()
    
    # Generate the rap song
    output_path = "output/hanumankind_rap.wav"
    print("Generating rap song... This may take a few minutes.")
    vocals_path = generator.generate_rap(lyrics, output_path)
    print(f"Vocals generated and saved to: {vocals_path}")

if __name__ == "__main__":
    main()
