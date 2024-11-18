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

def quantize_signal(signal, levels):
  min_val, max_val = np.min(signal), np.max(signal)
  delta = (max_val - min_val) / levels
  quantized_signal = np.zeros_like(signal)
  quantization_error = np.zeros_like(signal)
  encoded_signal = []
  interval_indices = []

  for i, sample in enumerate(signal):
    zone_index = int(np.floor((sample - min_val) / delta))
    zone_index = np.clip(zone_index, 0, levels - 1)
    midpoint = min_val + (zone_index + 0.5) * delta
    quantized_signal[i] = midpoint
    quantization_error[i] = midpoint - sample
    encoded_signal.append(f"{zone_index:0{int(np.log2(levels))}b}")
    interval_indices.append(int(zone_index) + 1)
  return quantized_signal, quantization_error, encoded_signal, interval_indices


def perform_quantization():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        levels_or_bits = int(levels_bits_entry.get())
        if quantization_choice.get() == 1:  # Levels
            levels = levels_or_bits
        else:  # Bits
            levels = 2 ** levels_or_bits
        signal = read_signal_from_file(file)
        quantized_signal, quantization_error, encoded_signal,interval_indices = quantize_signal(signal, levels)
        # Plot and display results
        plot_result(quantized_signal, "Quantized Signal")
        plot_result(quantization_error, "Quantization Error")
        plt.figure()
        plt.plot(encoded_signal, label="Encoded Signal")
        plt.title("Encoded Signal")
        plt.xlabel("Sample Index")
        plt.ylabel("Encoded Value")
        plt.show()
    else:
        print("Please select a signal file.")
    if quantization_choice.get() == 1:  # Levels
      QuantizationTest2("Quan2_Out.txt",interval_indices,encoded_signal,quantized_signal,quantization_error)
    else:
      QuantizationTest1("Quan1_Out.txt",encoded_signal,quantized_signal)

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

def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return
    print("QuantizationTest2 Test case passed successfully")
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
def open_task3():
    # Create new window for task 3
    task3_window = Toplevel(root)
    task3_window.geometry("500x300")
    task3_window.title(" Quantization")

    # Declare as global to use in other functions
    global levels_bits_entry, quantization_choice
    quantization_choice = IntVar(task3_window)  # Associate IntVar with task3_window
    Label(task3_window, text="Choose quantization input type:").pack()
    Radiobutton(task3_window, text="Number of Levels", variable=quantization_choice, value=1).pack()
    Radiobutton(task3_window, text="Number of Bits", variable=quantization_choice, value=2).pack()
    Label(task3_window, text="Enter levels or bits:").pack()
    levels_bits_entry = Entry(task3_window, width=20)
    levels_bits_entry.pack()
    quantize_button = Button(task3_window, text="Quantize Signal", command=perform_quantization)
    quantize_button.pack()
def fourier_transform(freq):
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    freq = int (freq)
    if fourier_choice.get() == 1:
        signal = read_signal_from_file(file)
        N = len(signal)
        X = np.zeros(N, complex)
        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        amp = np.sqrt((X.real * X.real) + (X.imag * X.imag))
        phase_shift = np.atan2(X.imag, X.real)
        with open("Output_Signal_DFT_A,Phase.txt", 'r') as f:
            lines = f.readlines()
            signal_data = [[float(x.rstrip('f')) for x in line.split()] for line in lines[3:]]
        output_amp = np.array([row[0] for row in signal_data])
        output_phase=np.array([row[1] for row in signal_data])
        if SignalComapreAmplitude(amp,output_amp)&SignalComaprePhaseShift(phase_shift,output_phase):
            print("Test Cases passed Successfully")
        else:
            print("Test Cases failed")
        fundamental_freq = 2 * np.pi * freq / N
        frequencies = [fundamental_freq * i for i in range(1, N + 1)]
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.stem(frequencies, amp)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('DFT')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.stem(frequencies, phase_shift)
        plt.xlabel('Frequency')
        plt.ylabel('Phase Shift')
        plt.grid(True)
        plt.show()
    else:
        with open(file, 'r') as f:
            lines = f.readlines()
            signal_data = [[float(x.rstrip('f')) for x in line.split()] for line in lines[3:]]
        amp = np.array([row[0] for row in signal_data])
        phase=np.array([row[1] for row in signal_data])
        N = len(amp)
        X = np.zeros(N, complex)
        freq =np.zeros(N, complex)
        real =amp*np.cos(phase)
        img = amp*np.sin(phase)
        X += real + 1j * img
        for k in range(N):
            for n in range(N):
                freq[k] += X[n] * np.exp(2j * np.pi * k * n / N)
            freq[k]/=N
        output_freq = read_signal_from_file("Output_Signal_IDFT.txt")
        if SignalComapreAmplitude(freq,output_freq):
            print("Test Cases passed Successfully")
        else:
            print("Test Cases failed")
        indices = list(range(N))
        frq = [int(x) for x in np.real(freq)]
        plt.stem(indices,frq)
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        plt.title('IDFT')
        plt.grid(True)
        plt.show()


def SignalComapreAmplitude(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i]-SignalOutput[i])>0.001:
                return False
        return True

def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            A=round(SignalInput[i])
            B=round(SignalOutput[i])
            if abs(A-B)>0.0001:
                return False
        return True


def open_task4():
    # Create new window for task 3
    task4window = Toplevel(root)
    task4window.geometry("500x300")
    task4window.title(" Fourier Transform")

    # Declare as global to use in other functions
    global sampling_freq, fourier_choice
    fourier_choice = IntVar(task4window)  # Associate IntVar with task4window

    # Input type selection for quantization (levels or bits)
    Label(task4window, text="Choose Fourier Transform  type:").pack()
    Radiobutton(task4window, text="DFT", variable=fourier_choice, value=1).pack()
    Radiobutton(task4window, text="IDFT", variable=fourier_choice, value=2).pack()


    Label(task4window, text="Enter Sampling Frequency:").pack()
    sampling_freq = Entry(task4window, width=20)
    sampling_freq.pack()

    # Button to start quantization process
    FT_button = Button(task4window, text="Calculate Signal", command=lambda :fourier_transform(sampling_freq.get()))
    FT_button.pack()
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
task3_button = Button(root, text="Calculate Quantization", command=open_task3)
task3_button.pack()
task4_button = Button(root, text="Fourier Transform", command=open_task4)
task4_button.pack()
root.mainloop()