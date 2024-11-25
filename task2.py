from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import shared as sh

def add_signals_from_file(file1, file2):
    signal1 = sh.read_signal_from_file(file1)
    signal2 = sh.read_signal_from_file(file2)
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for addition.")
    return signal1 + signal2

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
    sh.plot_result(result, 'Result of Addition')
    file_name1="Signal1+signal2.txt"
    file_name2="signal1+signal3.txt"
    if "Signal2.txt" in file2 :
        sh.SignalSamplesAreEqual(file_name1,[], result)
    else:
        sh.SignalSamplesAreEqual(file_name2, [], result)



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
    sh.plot_result(result, 'Result of Addition')
    file_name1="signal1-signal2.txt"
    file_name2="signal1-signal3.txt"
    if "Signal2.txt" in file2:
        sh.SignalSamplesAreEqual(file_name1,[], result)
    else:
        sh.SignalSamplesAreEqual(file_name2, [], result)


def perform_multiplication():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        constant = float(constant_entry.get())
        result = multiply_signal_with_constant(file, constant)
        sh.plot_result(result, 'Result of Multiplication')
    else:
        print("Please select a signal file.")

    file_name1 = "MultiplySignalByConstant-Signal1 - by 5.txt"
    file_name2 = "MultiplySignalByConstant-signal2 - by 10.txt"
    if "Signal1.txt" in file:
        sh.SignalSamplesAreEqual(file_name1, [], result)
    else:
        sh.SignalSamplesAreEqual(file_name2, [], result)


def perform_squaring():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        result = square_signal(file)
        sh.plot_result(result, 'Result of Squaring')
    else:
        print("Please select a signal file.")
    file_name1 = "Output squaring signal 1.txt"
    if "Signal1.txt" in file:
        sh.SignalSamplesAreEqual(file_name1, [], result)


def perform_normalization():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        normalize_type = normalize_choice.get()
        result = normalize_signal(file, normalize_type)
        sh.plot_result(result, 'Result of Normalization')
    else:
        print("Please select a signal file.")
    file_name1 = "normalize of signal 1 (from -1 to 1)-- output.txt"
    file_name2 = "normlize signal 2 (from 0 to 1 )-- output.txt"
    if "Signal1.txt" in file:
        sh.SignalSamplesAreEqual(file_name1, [], result)
    else:
        sh.SignalSamplesAreEqual(file_name2, [], result)

def perform_accumulation():
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file:
        result = accumulate_signals(file)
        sh.plot_result(result, 'Result of Accumulation')
    else:
        print("Please select a signal file.")
    file_name1 = "output accumulation for signal1.txt"
    if "Signal1.txt" in file:
        sh.SignalSamplesAreEqual(file_name1, [], result)



def add_signals(file, result):
    signal1 = sh.read_signal_from_file(file)
    if len(signal1) != len(result):
        raise ValueError("Signals must be of the same length for addition.")
    return signal1 + result

def subtract_signals_from_files(file1, file2):
    signal1 = sh.read_signal_from_file(file1)
    signal2 = sh.read_signal_from_file(file2)
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of the same length for subtraction.")
    return abs(signal1 - signal2)

def subtract_signals(file, result):
    signal1 = sh.read_signal_from_file(file)
    if len(signal1) != len(result):
        raise ValueError("Signals must be of the same length for subtraction.")
    return abs(result - signal1)

def multiply_signal_with_constant(file, constant):
    signal = sh.read_signal_from_file(file)
    return signal * constant

def square_signal(file):
    signal = sh.read_signal_from_file(file)
    return signal ** 2

def normalize_signal(file, normalize_type):
    signal = sh.read_signal_from_file(file)
    if normalize_type == 1:  # Normalize to 0 to 1
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    else:  # Normalize to -1 to 1
        return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

def accumulate_signals(file):
    signal = sh.read_signal_from_file(file)
    accumulated_signal = []
    total = 0
    for value in signal:
        total += value
        accumulated_signal.append(total)
    return accumulated_signal
def open_task2(root):
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
