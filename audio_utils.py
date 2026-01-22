"""
Audio Denoising Model - Helper Utilities
This module provides utility functions for audio denoising inference
"""

import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path


class AudioDenoiser:
    """Wrapper class for audio denoising inference"""
    
    def __init__(self, model_path='audio_denoiser_final.h5', sr=16000, n_fft=512, hop_length=128):
        """
        Initialize the denoiser
        
        Args:
            model_path: Path to trained model
            sr: Sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
        """
        self.model = tf.keras.models.load_model(model_path)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def denoise_audio(self, audio_path, output_path=None):
        """
        Denoise an audio file
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save denoised audio (optional)
            
        Returns:
            Denoised audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Get spectrogram and phase
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Denoise magnitude spectrogram
        mag_expanded = np.expand_dims(np.expand_dims(mag, 0), -1)
        denoised_mag = self.model.predict(mag_expanded, verbose=0)
        denoised_mag = np.squeeze(denoised_mag, (0, -1))
        
        # Reconstruct audio
        D_denoised = denoised_mag * np.exp(1j * phase)
        denoised_audio = librosa.istft(D_denoised, hop_length=self.hop_length)
        
        # Save if output path provided
        if output_path:
            import soundfile as sf
            sf.write(output_path, denoised_audio, self.sr)
        
        return denoised_audio
    
    def denoise_batch(self, audio_dir, output_dir):
        """
        Denoise multiple audio files in a directory
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save denoised files
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(audio_dir.glob('*.wav'))
        
        for audio_file in audio_files:
            output_file = output_dir / f"denoised_{audio_file.name}"
            self.denoise_audio(str(audio_file), str(output_file))
            print(f"✓ Processed: {audio_file.name}")


def evaluate_denoising(original_audio_path, denoised_audio_path):
    """
    Evaluate denoising quality
    
    Args:
        original_audio_path: Path to original clean audio
        denoised_audio_path: Path to denoised audio
        
    Returns:
        Dictionary with evaluation metrics
    """
    import soundfile as sf
    from pesq import pesq
    from pystoi import stoi
    
    # Load audio files
    original, sr_orig = sf.read(original_audio_path)
    denoised, sr_denoise = sf.read(denoised_audio_path)
    
    # Ensure same sample rate
    if sr_orig != sr_denoise:
        raise ValueError("Sample rates must match")
    
    # Trim to same length
    min_len = min(len(original), len(denoised))
    original = original[:min_len]
    denoised = denoised[:min_len]
    
    # Calculate metrics
    mse = np.mean((original - denoised) ** 2)
    mae = np.mean(np.abs(original - denoised))
    
    try:
        pesq_score = pesq(sr_orig, original, denoised, 'wb')
    except:
        pesq_score = None
    
    try:
        stoi_score = stoi(original, denoised, sr_orig)
    except:
        stoi_score = None
    
    return {
        'mse': mse,
        'mae': mae,
        'pesq': pesq_score,
        'stoi': stoi_score
    }


def compare_spectrograms(audio_path1, audio_path2, sr=16000, n_fft=512, hop_length=128):
    """
    Compare spectrograms of two audio files
    
    Args:
        audio_path1: Path to first audio file
        audio_path2: Path to second audio file
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        Tuple of magnitude spectrograms
    """
    audio1, _ = librosa.load(audio_path1, sr=sr)
    audio2, _ = librosa.load(audio_path2, sr=sr)
    
    S1 = np.abs(librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length))
    S2 = np.abs(librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length))
    
    return S1, S2


if __name__ == "__main__":
    print("Audio Denoising Utilities Loaded")
    print("Usage: from audio_utils import AudioDenoiser, evaluate_denoising")
