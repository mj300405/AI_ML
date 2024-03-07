import librosa
import numpy as np

class PianoAudioProcessor:
    def __init__(self):
        pass

    def load_audio(self, file_path):
        """Load an audio file with librosa."""
        signal, sr = librosa.load(file_path, sr=None)
        return signal, sr

    def extract_features(self, signal, sr):
        """Extract a comprehensive set of features for polyphonic piano transcription."""
        # Base features
        cqt = np.abs(librosa.cqt(signal, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12, tuning=None))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, sr=sr)
        harmonic, percussive = librosa.effects.hpss(signal)
        spectral_contrast = librosa.feature.spectral_contrast(S=cqt, sr=sr)
        delta_cqt = librosa.feature.delta(cqt)
        delta_chroma = librosa.feature.delta(chroma_cqt)

        # Additional features
        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
        chroma_stft = librosa.feature.chroma_stft(signal, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(signal, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)

        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=signal)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)

        return {
            'cqt': cqt_db,
            'chroma_cqt': chroma_cqt,
            'harmonic': harmonic,
            'percussive': percussive,
            'spectral_contrast': spectral_contrast,
            'delta_cqt': delta_cqt,
            'delta_chroma': delta_chroma,
            'mfcc': mfcc,
            'chroma_stft': chroma_stft,
            'chroma_cens': chroma_cens,
            'tonnetz': tonnetz,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'zero_crossing_rate': zero_crossing_rate
        }

    def preprocess(self, file_path):
        """Load an audio file and extract a comprehensive set of features optimized for polyphonic transcription."""
        signal, sr = self.load_audio(file_path)
        features = self.extract_features(signal, sr)
        return features
