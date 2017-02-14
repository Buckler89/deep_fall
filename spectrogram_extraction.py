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
        



###############################################################################################    
#    spec = []
#    longest = 0
#    for w in wav_filenames:
#        ###################### calcola gli spettri
#        if len(Sxx[1])>longest:
#            longest=len(Sxx[1])
#        spec.append([w, Sxx])
#
#
## zeropadda gli spettrogrammi salvati in source e li salva in dest
#def zeropads(source, dest):
#    
#
#
#
#
#
#for w in spec:
#    np.save(dest_path + w[0][0:-4],w[1],False)  
#    w[1]=np.lib.pad(w[1], ((0, 0), (0, longest-len(w[1][1]))),'constant', constant_values=(0, 0))
#    np.save(dest_path_zero_pad + w[0][0:-4],w[1],False)



if __name__ == "__main__":

    wav_dir_path = '/home/daniele/Scrivania/all_file/'
    dest_path='/home/daniele/Scrivania/spectrograms/'
    dest_path_zero_pad='//home/daniele/Scrivania/spectrograms_zero_pad/'
    window_type = 'hamming'
    fft_length = 256
    overlap = 128
    Fs = 8000    
    
    spectrograms(wav_dir_path, dest_path, Fs, fft_length, overlap, 'hamming')