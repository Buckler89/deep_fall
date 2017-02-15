#import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa
from os import walk, path

# estrae gli spettrogrammi dei file contenuti in source e li salva in dest
def spectrograms(source, dest, fs, N, overlap, win_type='hamming'):
    
    # list all file in source directory
    filenames = []
    for (dirpath, dirnames,  filenames) in walk(source):
        break
    # drop all non wav file
    wav_filenames = [f for f in filenames if f.lower().endswith('.wav')]

    for w in wav_filenames:
        # Load an audio file as a floating point time series
        x, fs  = librosa.core.load(path.join(source,w),sr=fs)
        # Returns: np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype], dtype=64-bit complex
        X = librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type,N), hop_length=N-overlap, center=False)
        #Sxx = np.abs(X)**2
        Sxx = librosa.logamplitude(np.abs(X)**2,ref_power=np.max)
        np.save(path.join(dest,w[0:-4]),Sxx)

def log_mel(source, dest, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True, delta_width=2):
    # with htk=True use HTK formula instead of Slaney
    # list all file in source directory
    filenames = []
    for (dirpath, dirnames,  filenames) in walk(source):
        break
    # drop all non wav file
    wav_filenames = [f for f in filenames if f.lower().endswith('.wav')]

    for w in wav_filenames:
        coefficients = []
        # Load an audio file as a floating point time series
        x, fs  = librosa.core.load(path.join(source,w),sr=fs)
        # Power spectrum
        S = np.abs(librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type,N), hop_length=N-overlap, center=False))**2
        # Build a Mel filter
        mel_basis = librosa.filters.mel(fs, N, n_mels, fmin, fmax, htk)
        # Filtering
        mel_filtered = np.dot(mel_basis, S)
        
        coefficients.append(librosa.logamplitude(mel_filtered))
        # add delta e delta-deltas
        coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1))
        coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=2, axis=-1))
        
        np.save(path.join(dest,w[0:-4]),coefficients)
    
if __name__ == "__main__":

    wav_dir_path = '/home/daniele/Scrivania/all_file/'
    dest_path='/home/daniele/Scrivania/spectrograms/'
    dest_path_log_mel='/home/daniele/Scrivania/logMels/'
    window_type = 'hamming'
    fft_length = 256
    overlap = 128
    Fs = 8000    
    n_mels=13
    fmin=0.0
    fmax=Fs/2
    htk=True
    delta_width=2
    
    spectrograms(wav_dir_path, dest_path, Fs, fft_length, overlap, 'hamming')
    log_mel(wav_dir_path, dest_path_log_mel, Fs, fft_length, overlap, window_type, n_mels, fmin, fmax, htk, delta_width)
