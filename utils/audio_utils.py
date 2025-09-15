import librosa
import numpy as np
import pyttsx3

def get_audio_energy(video_path):
    y, sr = librosa.load(video_path, sr=None)
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
    return energy

def generate_tts(text, output_path="audio/summary_voice.mp3"):
    tts_engine = pyttsx3.init()
    tts_engine.save_to_file(text, output_path)
    tts_engine.runAndWait()
    return output_path
