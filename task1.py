from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import shared as sh


def browse_file():
    filename = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if filename:
        plot_signal1(filename)


def plot_signal1(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        signal_data = [[float(x) for x in line.split()] for line in lines[3:]]
    time = [row[0] for row in signal_data]
    amplitude = [row[1] for row in signal_data]
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(time, amplitude, label='Continuous Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Continuous Signal')
    axs[1].stem(time, amplitude, label='Discrete Signal')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Discrete Signal')
    plt.tight_layout()
    plt.show()


def generate_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type):
    duration = 1
    num_samples = int(sampling_frequency)*duration
    num_samples -= 1
    t = np.linspace(0,  num_samples/sampling_frequency, int(sampling_frequency))
    if signal_type == "1":
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    elif signal_type == "2":
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal


def plot_signal(t, signal,amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type):
    #amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = main.parse_input()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].stem(t, signal, label="Discrete Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Discrete Signal")
    axs[0].set_xlim(0, 0.025)
    axs[1].plot(t, signal, label="Continuous Signal")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    if str(signal_type) == "1":
        axs[1].set_title("Continuous Signal(Sine Wave)")
    else:
        axs[1].set_title("Continuous Signal(Cosine Wave)")
    period = 1 / analog_frequency
    num_periods_to_show = 5
    axs[1].set_xlim(0, num_periods_to_show * period)
    plt.tight_layout()
    plt.show()


def generate_signal_and_compare(amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type):
#    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = main.parse_input()
    t, signal = generate_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type)
    plot_signal(t, signal,amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type)
    if signal_type == "1":
        file_name = "SinOutput.txt"
    else:
        file_name = "CosOutput.txt"
    if file_name:
        sh.SignalSamplesAreEqual(file_name,t, signal)
    else:
        print("Please select a valid text file for comparison.")

# task2
def generate_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type):
   # amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type = main.parse_input()
    num_samples = int(sampling_frequency)
    t = np.linspace(0, (num_samples - 1) / sampling_frequency, int(sampling_frequency))
    if signal_type == "1":
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    elif signal_type == "2":
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal


def plot_signal(t, signal,analog_frequency):
    #amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type = main.parse_input()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].stem(t, signal, label="Discrete Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Discrete Signal")
    axs[0].set_xlim(0, 0.025)
    axs[1].plot(t, signal, label="Continuous Signal")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Continuous Signal")
    period = 1 / analog_frequency
    num_periods_to_show = 5
    axs[1].set_xlim(0, num_periods_to_show * period)
    plt.tight_layout()
    plt.show()