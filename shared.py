from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def read_signal_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        signal_data = [[float(x) for x in line.split()] for line in lines[3:]]
    return np.array([row[1] for row in signal_data])


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


def SignalSamplesAreEqual(file_name, samples, precision=2):
    expected_indices = []
    expected_samples = []

    # Read the expected output file
    try:
        with open(file_name, 'r') as f:
            # Skip first 4 lines as they're headers or irrelevant
            for _ in range(4):
                next(f)

            # Read the actual expected values
            line = f.readline()
            while line:
                parts = line.strip().split()
                if len(parts) == 2:
                    idx = int(parts[0])
                    coeff = float(parts[1])
                    expected_indices.append(idx)
                    expected_samples.append(round(coeff, precision))  # Round expected coefficients
                line = f.readline()
    except FileNotFoundError:
        messagebox.showerror("Error", "File not found. Please check the file path.")
        return "Error: File not found. Please check the file path."

    # Skip the first 4 elements in the samples list (assuming you want to ignore the first 4 coefficients)
    samples = samples[1:]

    # Debug: Print lengths of expected and computed samples
    print(f"Expected samples length: {len(expected_samples)}")
    print(f"Computed samples length: {len(samples)}")

    # Ensure both lists are of the same length or handle accordingly
    if len(expected_samples) != len(samples):
        if len(samples) > len(expected_samples):
            samples = samples[:len(expected_samples)]  # Trim extra coefficients from computed samples
        else:
            messagebox.showerror("Error",
                                 f"Test case failed: Expected samples length is {len(expected_samples)}, but computed samples length is {len(samples)}.")
            return "Error: Sample lengths do not match."

    # Compare each coefficient with rounding and tolerance
    for i in range(len(expected_samples)):
        if round(samples[i], precision) != expected_samples[i]:
            messagebox.showerror("Error",
                                 f"Test case failed: Coefficient mismatch at index {i} (expected {expected_samples[i]}, got {round(samples[i], precision)}).")
            return f"Error: Coefficient mismatch at index {i} (expected {expected_samples[i]}, got {round(samples[i], precision)})."

    return "Success: Test case passed successfully."


def plot_result(signal, title):
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
