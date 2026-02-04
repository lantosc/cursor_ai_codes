"""
Audio Analysis Module for extracting visemes and speech features
"""
import numpy as np
import librosa
import os

class AudioAnalyzer:
    """Analyze audio to extract visemes (mouth shapes) for lip-sync"""
    
    # Viseme mapping: phonemes to mouth shapes
    VISEME_MAP = {
        # Closed/rest position
        'silence': 0,
        # Wide open (A, E, I)
        'A': 1, 'E': 1, 'I': 1, 'AE': 1,
        # Round open (O, U, W)
        'O': 2, 'U': 2, 'W': 2, 'AW': 2,
        # Closed lips (M, B, P)
        'M': 3, 'B': 3, 'P': 3,
        # Fricatives (F, V)
        'F': 4, 'V': 4,
        # Th sounds
        'TH': 5,
        # Open teeth (D, T, N, L)
        'D': 6, 'T': 6, 'N': 6, 'L': 6,
        # Tongue (S, Z, CH, SH)
        'S': 7, 'Z': 7, 'CH': 7, 'SH': 7,
        # Open jaw (G, K, R)
        'G': 8, 'K': 8, 'R': 8,
    }
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_time = self.hop_length / self.sample_rate  # ~0.023 seconds per frame
    
    def analyze_audio(self, audio_path):
        """
        Analyze audio file and extract viseme sequence
        Returns: list of (time, viseme_id, intensity) tuples
        """
        if not os.path.exists(audio_path):
            return []
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            # 1. Energy (loudness) - indicates speech activity
            energy = np.abs(librosa.stft(y, hop_length=self.hop_length))
            energy = np.mean(energy, axis=0)
            
            # 2. Zero crossing rate (ZCR) - indicates fricatives
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            
            # 3. Spectral centroid - indicates vowel quality
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            
            # 4. MFCC (Mel-frequency cepstral coefficients) - speech characteristics
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Normalize features
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
            zcr = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-6)
            spectral_centroid = (spectral_centroid - spectral_centroid.min()) / (spectral_centroid.max() - spectral_centroid.min() + 1e-6)
            
            # Generate viseme sequence based on audio features
            visemes = []
            n_frames = len(energy)
            
            for i in range(n_frames):
                time_pos = i * self.frame_time
                
                # Determine viseme based on features
                if energy[i] < 0.1:  # Silence
                    viseme_id = 0
                    intensity = 0.0
                elif spectral_centroid[i] > 0.6:  # High frequency = vowels (A, E, I)
                    viseme_id = 1  # Wide open
                    intensity = energy[i]
                elif spectral_centroid[i] < 0.3:  # Low frequency = O, U
                    viseme_id = 2  # Round open
                    intensity = energy[i]
                elif zcr[i] > 0.5:  # High ZCR = fricatives
                    viseme_id = 4  # F, V
                    intensity = min(energy[i] * 1.5, 1.0)
                elif energy[i] > 0.7:  # High energy = open mouth
                    viseme_id = 1  # Wide open
                    intensity = energy[i]
                else:  # Medium energy = neutral
                    viseme_id = 0  # Rest position
                    intensity = energy[i] * 0.5
                
                visemes.append((time_pos, viseme_id, intensity))
            
            return visemes
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return []
    
    def get_viseme_at_time(self, viseme_sequence, time):
        """Get viseme at a specific time"""
        if not viseme_sequence:
            return (0, 0.0)  # Default: rest position
        
        # Find closest viseme
        closest = min(viseme_sequence, key=lambda x: abs(x[0] - time))
        return (closest[1], closest[2])  # (viseme_id, intensity)
