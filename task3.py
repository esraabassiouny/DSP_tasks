from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import shared as sh

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


def perform_quantization():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        levels_or_bits = int(levels_bits_entry.get())
        if quantization_choice.get() == 1:  # Levels
            levels = levels_or_bits
        else:  # Bits
            levels = 2 ** levels_or_bits
        signal = sh.read_signal_from_file(file)
        quantized_signal, quantization_error, encoded_signal,interval_indices = quantize_signal(signal, levels)
        # Plot and display results
        sh.plot_result(quantized_signal, "Quantized Signal")
        sh.plot_result(quantization_error, "Quantization Error")
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


def open_task3(root):
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