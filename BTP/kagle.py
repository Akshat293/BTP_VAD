import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools

INPUT_DIR = '/Users/akshatsaxena/Downloads/free-spoken-digit-dataset-master/recordings/'
OUTPUT_DIR = '/Users/akshatsaxena/Downloads/free-spoken-digit-dataset-master/spectrograms/'


parent_list = os.listdir(INPUT_DIR)
# for i in range(10):
#     print(parent_list[i])


# for i in range(5): 
#     signal_wave = wave.open(os.path.join(INPUT_DIR, parent_list[i]), 'r')
#     sample_rate = 16000
#     sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)

#     plt.figure(figsize=(12,12))
#     plot_a = plt.subplot(211)
#     plot_a.set_title(parent_list[i])
#     plot_a.plot(sig)
#     plot_a.set_xlabel('sample rate * time')
#     plot_a.set_ylabel('energy')

#     plot_b = plt.subplot(212)
#     plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
#     plot_b.set_xlabel('Time')
#     plot_b.set_ylabel('Frequency')

# plt.show()

# Utility function to get sound and frame rate info
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# For every recording, make a spectogram and save it as label_speaker_no.png
if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))
    
for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)
            file_stem = Path(file_path).stem
            sound_info, frame_rate = get_wav_info(file_path)
            pylab.specgram(sound_info, Fs=frame_rate)
            pylab.savefig(f'{file_dist_path}.png')
            pylab.close()

# Print the ten classes in our dataset
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images'))
print("Classes: \n")
for i in range(10):
    print(path_list[i])
    
# File names for class 1
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images/class_1'))
print("\nA few example files: \n")
for i in range(10):
    print(path_list[i])