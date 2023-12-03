import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyaudio
import wave
import numpy as np
import torch

torch.set_num_threads(1)


def record_audio(duration, output_file):
    format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    chunk = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording audio...")

    frames = []

    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()


def main():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    sampling_rate = 16000

    wav = read_audio('recorded_audio.wav')

    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)

    if speech_timestamps:
        print(speech_timestamps)
        save_audio('speech.wav', collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
    else:
        print("No speech detected in the audio.")


class AudioRecorder(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Audio Recorder and Processor')
        self.setGeometry(100, 100, 800, 600)

        self.duration_label = QLabel('Recording Duration (seconds):', self)
        self.duration_input = QLineEdit(self)
        self.duration_input.setValidator(QtGui.QIntValidator())  # Only allow integer input

        self.record_button = QPushButton('Record Audio', self)
        self.record_button.clicked.connect(self.recordAudio)

        self.process_button = QPushButton('Process Audio', self)
        self.process_button.clicked.connect(self.processAudio)

        self.display_button = QPushButton('Display Waveforms', self)
        self.display_button.clicked.connect(self.displayWaveforms)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.resetUI)

        self.record_status_label = QLabel('Recording status: Not started', self)
        self.process_status_label = QLabel('Processing status: Not started', self)

        self.figure, self.ax = plt.subplots(2, 1, figsize=(6, 4))

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        layout = QVBoxLayout()
        layout.addWidget(self.duration_label)
        layout.addWidget(self.duration_input)
        layout.addWidget(self.record_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.display_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.record_status_label)
        layout.addWidget(self.process_status_label)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def recordAudio(self):
        duration_text = self.duration_input.text()
        if duration_text.isnumeric():
            duration = int(duration_text)
            output_file = "recorded_audio.wav"

            self.record_status_label.setText('Recording status: In progress...')
            record_audio(duration, output_file)
            self.record_status_label.setText('Recording status: Completed')
        else:
            self.record_status_label.setText('Invalid duration. Please enter a numeric value.')

    def processAudio(self):
        self.process_status_label.setText('Processing status: In progress...')
        main()
        self.process_status_label.setText('Processing status: Completed')

    def displayWaveforms(self):
        self.ax[0].clear()
        self.ax[1].clear()

        audio1 = wave.open('recorded_audio.wav', 'r')
        audio2 = wave.open('speech.wav', 'r')

        frames1 = audio1.getnframes()
        frames2 = audio2.getnframes()

        channels1 = audio1.getnchannels()
        channels2 = audio2.getnchannels()

        sample_rate1 = audio1.getframerate()
        sample_rate2 = audio2.getframerate()

        signal1 = audio1.readframes(frames1)
        signal2 = audio2.readframes(frames2)

        signal1 = np.frombuffer(signal1, dtype=np.int16)
        signal2 = np.frombuffer(signal2, dtype=np.int16)

        signal1 = np.reshape(signal1, (frames1, channels1))
        signal2 = np.reshape(signal2, (frames2, channels2))

        time1 = np.linspace(0, len(signal1) / sample_rate1, num=len(signal1))
        time2 = np.linspace(0, len(signal2) / sample_rate2, num=len(signal2))

        self.ax[0].plot(time1, signal1)
        self.ax[0].set_title('Original Audio')
        self.ax[0].set_xlabel('Time (seconds)')
        self.ax[0].set_ylabel('Amplitude')

        self.ax[1].plot(time2, signal2)
        self.ax[1].set_title('Speech Audio')
        self.ax[1].set_xlabel('Time (seconds)')
        self.ax[1].set_ylabel('Amplitude')

        self.canvas.draw()

        audio1.close()
        audio2.close()

        os.remove('recorded_audio.wav')
        os.remove('speech.wav')

    def resetUI(self):
        # Reset all labels and clear plots
        self.duration_input.clear()
        self.record_status_label.clear()
        self.process_status_label.clear()
        self.ax[0].clear()
        self.ax[1].clear()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    ex.show()
    sys.exit(app.exec_())
