from tkinter import *
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import os

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


def browse_file():
    filename = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if filename:
        plot_signal1(filename)


def generate_signal():
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = parse_input()
    if sampling_frequency < 2 * analog_frequency:
        raise ValueError("Sampling frequency must be at least twice the analog frequency.")
    num_samples = int(sampling_frequency)
    t = np.linspace(0,  (num_samples-1)/sampling_frequency, int(sampling_frequency))

    if signal_type == "1":
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    elif signal_type == "2":
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal



def plot_signal(t, signal):
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = parse_input()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # Plot the discrete signal
    axs[0].stem(t, signal, label="Discrete Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Discrete Signal")
    axs[0].set_xlim(0, 0.025)
    # Plot the continuous signal
    axs[1].plot(t, signal, label="Continuous Signal")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Continuous Signal")
    period = 1 / analog_frequency
    num_periods_to_show = 5
    axs[1].set_xlim(0, num_periods_to_show * period)
    plt.tight_layout()
    plt.show()


def parse_input():
    amp = float(ampValue.get())
    theta = float(thetaValue.get())
    freq = float(freqValue.get())
    sampling_freq = float(samplingFreqValue.get())
    type = str(funct.get())
    return amp,theta ,freq,sampling_freq, type


def SignalSamplesAreEqual(file_name, indices,samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    if len(expected_samples) != len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")

def generate_signal_and_compare():
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = parse_input()
    t, signal = generate_signal()
    plot_signal(t, signal)
    if signal_type == "1":
        file_name = "SinOutput.txt"
    else:
        file_name = "CosOutput.txt"
    if file_name:
        SignalSamplesAreEqual(file_name,t, signal)
    else:
        print("Please select a valid text file for comparison.")


# gui
root = Tk()
root.geometry("500x400")
funct = IntVar()
browse_button = Button(root, text="Browse File", command=browse_file)
browse_button.grid()
sinbutton = Radiobutton(root,text="sin",value=1,variable=funct)
cosbutton = Radiobutton(root,text="cos",value=2,variable=funct)
sinbutton.grid()
cosbutton.grid()
ampLabel =Label(root,text="Amplitude")
ampLabel.grid(row=4,column=0)
ampValue = Entry(root,width=20)
ampValue.grid(row = 4,column=1)
thetaLabel =Label(root,text="Phase Shift")
thetaLabel.grid(row=6,column=0)
thetaValue = Entry(root,width=20)
thetaValue.grid(row = 6,column=1)
freqLabel =Label(root,text="Frequency")
freqLabel.grid(row=8,column=0)
freqValue = Entry(root,width=20)
freqValue.grid(row = 8,column=1)
samplingFreqLabel =Label(root,text="Sampling Frequency")
samplingFreqLabel.grid(row=10,column=0)
samplingFreqValue = Entry(root,width=20)
samplingFreqValue.grid(row =10,column=1)
generateButton = Button(root,text="Generate Signal",command=lambda: generate_signal_and_compare())
generateButton.grid()
root.mainloop()