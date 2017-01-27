import numpy as np
from scipy import signal
from scipy.io import wavfile
#import matplotlib.pyplot as plt
from os import walk



####################### da spostare sul file di configurazione dell'esperimento

wav_dir_path = '/media/buckler/DataSSD/Phd/fall_detection/dataset/all_file/'
dest_path='/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrogram_zero_pad/'
window_type = 'hamming'
window_length = 256
overlap = 128



# all file in wav_dir_path directory  <-----TODO subdie & drop from list no wav file 
wav_filenames = []
for (dirpath, dirnames,  wav_filenames) in walk(wav_dir_path):
    break

spec = []
longest = 0
for w in wav_filenames:
    ###################### calcola gli spettri
    [fs,x] = wavfile.read(dirpath + "/" + w)
    f, t, Sxx = signal.spectrogram(x, fs, window_type, window_length, overlap)
    if len(Sxx[1])>longest:
        longest=len(Sxx[1])
    spec.append([w, Sxx])

###################### uniforma il vettore degli spettri: zero padding

for w in spec:
    w[1]=np.lib.pad(w[1], ((0, 0), (0, longest-len(w[1][1]))),'constant', constant_values=(0, 0))
    #np.save(dest_path + w[0][0:-4],w[1],False)
    

############################## salva tutto
# x=np.load(filename) to load
#np.save('data',spec,False)




#plt.pcolormesh(t, f, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
#from os import listdir
#from os.path import isfile, join
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


