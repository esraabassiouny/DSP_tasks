from tkinter import *
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk



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


def generate_signal():
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = parse_input()
    duration = 1
    num_samples = int(sampling_frequency)*duration
    num_samples -= 1
    t = np.linspace(0,  num_samples/sampling_frequency, int(sampling_frequency))
    if signal_type == "1":
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    elif signal_type == "2":
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal



def plot_signal(t, signal):
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type, = parse_input()
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

# task2
def generate_signal():
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type = parse_input()
    num_samples = int(sampling_frequency)
    t = np.linspace(0, (num_samples - 1) / sampling_frequency, int(sampling_frequency))
    if signal_type == "1":
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    elif signal_type == "2":
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal
def read_signal_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        signal_data = [[float(x) for x in line.split()] for line in lines[3:]]
    return np.array([row[1] for row in signal_data])

def add_signals_from_file(file1, file2):
    signal1 = read_signal_from_file(file1)
    signal2 = read_signal_from_file(file2)
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for addition.")
    return signal1 + signal2

def add_signals(file, result):
    signal1 = read_signal_from_file(file)
    if len(signal1) != len(result):
        raise ValueError("Signals must be of the same length for addition.")
    return signal1 + result

def subtract_signals_from_files(file1, file2):
    signal1 = read_signal_from_file(file1)
    signal2 = read_signal_from_file(file2)
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for subtraction.")
    return abs(signal1 - signal2)

def subtract_signals(file, result):
    signal1 = read_signal_from_file(file)
    if len(signal1) != len(result):
        raise ValueError("Signals must be of the same length for subtraction.")
    return abs(result - signal1)

def multiply_signal_with_constant(file, constant):
    signal = read_signal_from_file(file)
    return signal * constant

def square_signal(file):
    signal = read_signal_from_file(file)
    return signal ** 2

def normalize_signal(file, normalize_type):
    signal = read_signal_from_file(file)
    if normalize_type == 1:  # Normalize to 0 to 1
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    else:  # Normalize to -1 to 1
        return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

def accumulate_signals(file):
    signal = read_signal_from_file(file)
    accumulated_signal = []
    total = 0
    for value in signal:
        total += value
        accumulated_signal.append(total)
    return accumulated_signal

def plot_result(signal, title):
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

def perform_addition(no_of_signals):
    no_of_signals = int(no_of_signals)
    no_of_signals-=2
    file1 = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    file2 = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file1 and file2:
        result = add_signals_from_file(file1, file2)
    else:
        print("Please select both signal files.")
    for i in range(no_of_signals):
        file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        result = add_signals(file, result)
    plot_result(result, 'Result of Addition')
    file_name1="Signal1+signal2.txt"
    file_name2="signal1+signal3.txt"
    if "Signal2.txt" in file2 :
        SignalSamplesAreEqual(file_name1,[], result)
    else:
        SignalSamplesAreEqual(file_name2, [], result)



def perform_subtraction(no_of_signals):
    no_of_signals = int(no_of_signals)
    no_of_signals -= 2
    file1 = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    file2 = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file1 and file2:
        result = subtract_signals_from_files(file1, file2)
    else:
        print("Please select both signal files.")
    for i in range(no_of_signals):
        file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        result = subtract_signals(file, result)
    plot_result(result, 'Result of Addition')
    file_name1="signal1-signal2.txt"
    file_name2="signal1-signal3.txt"
    if "Signal2.txt" in file2:
        SignalSamplesAreEqual(file_name1,[], result)
    else:
        SignalSamplesAreEqual(file_name2, [], result)


def perform_multiplication():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        constant = float(constant_entry.get())
        result = multiply_signal_with_constant(file, constant)
        plot_result(result, 'Result of Multiplication')
    else:
        print("Please select a signal file.")

    file_name1 = "MultiplySignalByConstant-Signal1 - by 5.txt"
    file_name2 = "MultiplySignalByConstant-signal2 - by 10.txt"
    if "Signal1.txt" in file:
        SignalSamplesAreEqual(file_name1, [], result)
    else:
        SignalSamplesAreEqual(file_name2, [], result)


def perform_squaring():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        result = square_signal(file)
        plot_result(result, 'Result of Squaring')
    else:
        print("Please select a signal file.")
    file_name1 = "Output squaring signal 1.txt"
    if "Signal1.txt" in file:
        SignalSamplesAreEqual(file_name1, [], result)


def perform_normalization():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        normalize_type = normalize_choice.get()
        result = normalize_signal(file, normalize_type)
        plot_result(result, 'Result of Normalization')
    else:
        print("Please select a signal file.")
    file_name1 = "normalize of signal 1 (from -1 to 1)-- output.txt"
    file_name2 = "normlize signal 2 (from 0 to 1 )-- output.txt"
    if "Signal1.txt" in file:
        SignalSamplesAreEqual(file_name1, [], result)
    else:
        SignalSamplesAreEqual(file_name2, [], result)

def perform_accumulation():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        result = accumulate_signals(file)
        plot_result(result, 'Result of Accumulation')
    else:
        print("Please select a signal file.")
    file_name1 = "output accumulation for signal1.txt"
    if "Signal1.txt" in file:
        SignalSamplesAreEqual(file_name1, [], result)


def plot_signal(t, signal):
    amplitude, phase_shift, analog_frequency, sampling_frequency, signal_type = parse_input()
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


def open_task2():
    task2_window = Toplevel(root)
    task2_window.geometry("500x400")
    task2_window.title("Task 2 - Arithmetic Operations")
    label = tk.Label(task2_window, text="Enter number of signals:")
    label.pack()
    entry = Entry(task2_window)
    entry.pack()
    add_button = Button(task2_window, text="Addition", command=lambda :perform_addition(entry.get()))
    add_button.pack()

    # Load files for Subtraction
    sub_button = Button(task2_window, text="Subtraction", command=lambda :perform_subtraction(entry.get()))
    sub_button.pack()

    # Multiplication with constant input
    global constant_entry
    constant_label = Label(task2_window, text="Enter constant for multiplication:")
    constant_label.pack()
    constant_entry = Entry(task2_window, width=20)
    constant_entry.pack()
    mul_button = Button(task2_window, text="Multiply", command=perform_multiplication)
    mul_button.pack()

    # Squaring
    square_button = Button(task2_window, text="Squaring", command=perform_squaring)
    square_button.pack()

    # Normalization with choice
    global normalize_choice
    normalize_choice = IntVar()
    normalize_label = Label(task2_window, text="Choose normalization type:")
    normalize_label.pack()
    radio_01 = Radiobutton(task2_window, text="Normalize from 0 to 1", variable=normalize_choice, value=1)
    radio_01.pack()
    radio_neg1_1 = Radiobutton(task2_window, text="Normalize from -1 to 1", variable=normalize_choice, value=2)
    radio_neg1_1.pack()
    normalize_button = Button(task2_window, text="Normalize", command=perform_normalization)
    normalize_button.pack()

    accumulate_button = Button(task2_window, text="Accumulation", command=perform_accumulation)
    accumulate_button.pack()

# gui
root = tk.Tk()
root.title("Signal Generator")
root.geometry("500x400")
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)
browse_button = ttk.Button(frame, text="Open Signal File", command=browse_file)
browse_button.grid(row=0, column=0, sticky="w")
funct = IntVar()
sinbutton = ttk.Radiobutton(frame, text="sin", value=1, variable=funct)
cosbutton = ttk.Radiobutton(frame, text="cos", value=2, variable=funct)
sinbutton.grid(row=1, column=0, sticky="w")
cosbutton.grid(row=2, column=0, sticky="w")
ampLabel = ttk.Label(frame, text="Amplitude")
ampLabel.grid(row=3, column=0, sticky="w")
ampValue = ttk.Entry(frame, width=20)
ampValue.grid(row=3, column=1, sticky="w")
thetaLabel = ttk.Label(frame, text="Phase Shift")
thetaLabel.grid(row=4, column=0, sticky="w")
thetaValue = ttk.Entry(frame, width=20)
thetaValue.grid(row=4, column=1, sticky="w")
freqLabel = ttk.Label(frame, text="Frequency")
freqLabel.grid(row=5, column=0, sticky="w")
freqValue = ttk.Entry(frame, width=20)
freqValue.grid(row=5, column=1, sticky="w")
samplingFreqLabel = ttk.Label(frame, text="Sampling Frequency")
samplingFreqLabel.grid(row=6, column=0, sticky="w")
samplingFreqValue = ttk.Entry(frame, width=20)
samplingFreqValue.grid(row=6, column=1, sticky="w")
generateButton = ttk.Button(frame, text="Generate Signal", command=generate_signal_and_compare)
generateButton.grid(row=7, column=0, columnspan=2, sticky="e")
task2_button = Button(text="Arithmetic Operations",command=open_task2)
task2_button.pack()
root.mainloop()