import math
from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from matplotlib import pyplot as plt

def load_files():
    global signals
    file_paths = filedialog.askopenfilenames(
        title="Select Signal Files",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not file_paths:
        return None

    signals = []

    try:
        for file_path in file_paths:
            indices = []
            samples = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    values = line.strip().split()
                    if len(values) == 2:
                        indices.append(int(values[0]))
                        samples.append(float(values[1]))
                signals.append({"indices": indices, "samples": samples})

        print("Signal loaded successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"Error loading files: {e}")

def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
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
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")
def perform_convolution(filter_type_var):
    global signals, filtered_signal, convolution_result
    if not signals:
        messagebox.showerror("Error", "No signals loaded.")
        return

    if not filtered_signal:
        messagebox.showerror("Error", "No filter has been applied. Please design and apply a filter first.")
        return

    try:
        # Input signal
        input_signal = signals[0]
        input_indices = input_signal['indices']
        input_samples = input_signal['samples']

        # Filtered signal
        filtered_indices = filtered_signal['indices']
        filtered_samples = filtered_signal['samples']

        # Convolution computation
        conv_result = np.convolve(input_samples, filtered_samples)
        start_index = input_indices[0] + filtered_indices[0]
        end_index = start_index + len(conv_result) - 1

        # Save convolution result
        convolution_result = {
            "indices": list(range(start_index, end_index + 1)),
            "samples": conv_result
        }
        # Get the selected filter type
        filter_type = filter_type_var.get()

        if filter_type == 'Low pass':
            Compare_Signals("ecg_low_pass_filtered.txt", convolution_result['indices'], convolution_result['samples'])
            print("ecg_low_pass_filtered Done")
        elif filter_type == 'High pass':
            Compare_Signals("ecg_high_pass_filtered.txt", convolution_result['indices'], convolution_result['samples'])
            print("ecg_high_pass_filtered Done")

        elif filter_type == 'Band pass':
            Compare_Signals("ecg_band_pass_filtered.txt", convolution_result['indices'], convolution_result['samples'])
            print("ecg_band_pass_filtered Done ")

        elif filter_type == 'Band stop':
            Compare_Signals("ecg_band_stop_filtered.txt", convolution_result['indices'], convolution_result['samples'])
            print("ecg_band_stop_filtered Done ")

    except Exception as e:
        messagebox.showerror("Error", f"Error performing convolution: {e}")
        print(f"Error performing convolution: {e}")

def test_FIR_filter(filter_type, fs, fc, fc2, stop_band_attenuation, transition_band):
    global filtered_signal
    try:
        indices, coefficients = FIR_filter_design(filter_type, fs, stop_band_attenuation, fc, transition_band, fc2)

        if len(indices) == 0 or len(coefficients) == 0:
            messagebox.showerror("Error", "Filter coefficients are empty. Please check your input values.")
            return

        filtered_signal = {'indices': np.array(indices), 'samples': np.array(coefficients)}


        if filter_type == 'Low pass':
            Compare_Signals("LPFCoefficients.txt", filtered_signal['indices'], filtered_signal['samples'])
        elif filter_type == 'High pass':
             Compare_Signals("HPFCoefficients.txt", filtered_signal['indices'], filtered_signal['samples'])
        elif filter_type == 'Band pass':
             Compare_Signals("BPFCoefficients.txt", filtered_signal['indices'], filtered_signal['samples'])
        elif filter_type == 'Band stop':
             Compare_Signals("BSFCoefficients.txt", filtered_signal['indices'], filtered_signal['samples'])

    except ValueError as e:
        messagebox.showerror("Error", f"Error designing filter: {e}")
        print(f"Error designing filter: {e}")
def FIR_filter_design(filter_type, fs, stop_band_attenuation, fc, transition_band, fc2=None):
    def window_function(stop_band_attenuation, n, N):
        if stop_band_attenuation <= 21:
            return 1
        elif stop_band_attenuation <= 44:
            return 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
        elif stop_band_attenuation <= 53:
            return 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
        elif stop_band_attenuation <= 74:
            return 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        else:
            return 1

    def round_up_to_odd(number):
        rounded_number = math.ceil(number)
        if rounded_number % 2 == 0:
            rounded_number += 1
        return rounded_number

    delta_f = transition_band / fs
    if stop_band_attenuation <= 21:
        N = round_up_to_odd(0.9 / delta_f)
    elif stop_band_attenuation <= 44:
        N = round_up_to_odd(3.1 / delta_f)
    elif stop_band_attenuation <= 53:
        N = round_up_to_odd(3.3 / delta_f)
    elif stop_band_attenuation <= 74:
        N = round_up_to_odd(5.5 / delta_f)

    h = []
    indices = range(-math.floor(N / 2), math.floor(N / 2) + 1)

    if filter_type == 'Low pass':
        new_fc = fc + 0.5 * transition_band
        new_fc /= fs
        for n in indices:
            w_n = window_function(stop_band_attenuation, n, N)
            h_d = 2 * new_fc if n == 0 else 2 * new_fc * (np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
            h.append(h_d * w_n)

    elif filter_type == 'High pass':
        new_fc = fc - 0.5 * transition_band
        new_fc /= fs
        for n in indices:
            w_n = window_function(stop_band_attenuation, n, N)
            h_d = 1 - 2 * new_fc if n == 0 else -2 * new_fc * (
                        np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
            h.append(h_d * w_n)

    elif filter_type == 'Band pass' and fc2 is not None:
        new_fc = fc - 0.5 * transition_band
        new_fc /= fs
        new_fc2 = fc2 + 0.5 * transition_band
        new_fc2 /= fs
        for n in indices:
            w_n = window_function(stop_band_attenuation, n, N)
            if n == 0:
                h_d = 2 * (new_fc2 - new_fc)
            else:
                h_d = (2 * new_fc2 * np.sin(n * 2 * np.pi * new_fc2) / (n * 2 * np.pi * new_fc2)) - \
                      (2 * new_fc * np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc))
            h.append(h_d * w_n)

    elif filter_type == 'Band stop' and fc2 is not None:
        new_fc = fc + 0.5 * transition_band
        new_fc /= fs
        new_fc2 = fc2 - 0.5 * transition_band
        new_fc2 /= fs
        for n in indices:
            w_n = window_function(stop_band_attenuation, n, N)
            if n == 0:
                h_d = 1 - 2 * (new_fc2 - new_fc)
            else:
                h_d = (2 * new_fc * np.sin(n * 2 * np.pi * new_fc) / (n * 2 * np.pi * new_fc)) - \
                      (2 * new_fc2 * np.sin(n * 2 * np.pi * new_fc2) / (n * 2 * np.pi * new_fc2))
            h.append(h_d * w_n)

    return indices, h
def open_filtering_popup(root):
    popup = tk.Toplevel(root)
    popup.title("FIR Filtering Options")
    popup.geometry("500x400")

    # Title Label
    tk.Label(popup, text="FIR Filter Designer", font=(16)).pack(pady=10)

    # Frame for Inputs
    input_frame = tk.Frame(popup)
    input_frame.pack(pady=10, padx=20, fill="x")

    # Helper function to create labeled input fields
    def create_input(parent, label_text, var_type, row):
        tk.Label(parent, text=label_text, font=("Arial", 12)).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry_var = var_type()
        entry = tk.Entry(parent, textvariable=entry_var, font=("Arial", 12), width=25)
        entry.grid(row=row, column=1, padx=10, pady=5)
        return entry_var

    # Inputs for filter parameters
    fs_var = create_input(input_frame, "FS (Hz):", tk.DoubleVar, 0)
    fc_var = create_input(input_frame, "Cutoff Frequency 1 (Hz):", tk.DoubleVar, 1)
    fc2_var = create_input(input_frame, "Cutoff Frequency 2 (optional):", tk.StringVar, 2)
    transition_band_var = create_input(input_frame, "Transition Bandwidth (Hz):", tk.DoubleVar, 3)
    delta_s_var = create_input(input_frame, "Stopband Attenuation (dB):", tk.DoubleVar, 4)

    # Dropdown for Filter Type
    tk.Label(input_frame, text="Filter Type:", font=("Arial", 12)).grid(row=5, column=0, padx=10, pady=5, sticky="w")
    filter_type_var = tk.StringVar(value="Low pass")
    filter_type_menu = ttk.Combobox(input_frame, textvariable=filter_type_var, values=["Low pass", "High pass", "Band pass", "Band stop"], state="readonly", width=22)
    filter_type_menu.grid(row=5, column=1, padx=10, pady=5)

    # Function to validate inputs and apply filter
    def apply_filter():
        try:
            # Get filter type and validate inputs
            filter_type = filter_type_var.get()
            fs = float(fs_var.get())
            fc = float(fc_var.get())
            fc2 = fc2_var.get().strip()

            if filter_type in ["Band pass", "Band stop"]:
                if not fc2:
                    messagebox.showerror("Error", "Second cutoff frequency is required for bandpass/bandstop filters.")
                    return
                fc2 = float(fc2)
            else:
                fc2 = None

            stop_band_attenuation = float(delta_s_var.get())
            transition_band = float(transition_band_var.get())

            test_FIR_filter(filter_type, fs, fc, fc2, stop_band_attenuation, transition_band)

            messagebox.showinfo("Success", f"{filter_type} filter designed successfully.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    # Function to plot loaded signal
    def plot_signal():
        try:
            if 'signals' not in globals() or not signals:
                messagebox.showerror("Error", "No signals loaded. Please load a signal first.")
                return

            signal_data = signals[0]['samples']  # Assuming first signal for simplicity
            fs = float(fs_var.get())  # Sampling frequency

            time = np.arange(len(signal_data)) / fs
            plt.figure(figsize=(10, 6))
            plt.plot(time, signal_data, label="Loaded Signal")
            plt.title("Loaded Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    # Buttons
    button_frame = tk.Frame(popup)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Load Signals", font=("Arial", 12), command=load_files).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Apply Filter", font=("Arial", 12), command=apply_filter).grid(row=0, column=1, padx=10)
    tk.Button(button_frame, text="convolve", font=("Arial", 12),
              command=lambda: perform_convolution(filter_type_var),).grid(row=0, column=2, padx=10)
    tk.Button(button_frame, text="Cancel", font=("Arial", 12), command=popup.destroy).grid(row=0, column=3, padx=10)

def resample_signal(M_var, L_var, fs_var, fc_var, transition_band_var, delta_s_var):
    try:
        M = M_var.get()
        L = L_var.get()
        fs = fs_var.get()
        fc = fc_var.get()
        transition_band = transition_band_var.get()
        stop_band_attenuation = delta_s_var.get()

        # Validate that no fields are empty or None
        if M == "" or L == "" or fs == "" or fc == "" or transition_band == "" or stop_band_attenuation == "":
            messagebox.showerror("Error", "All fields are required. Please fill in all fields.")
            return

        try:
            M = int(M)
            L = int(L)
            fs = float(fs)
            fc = float(fc)
            transition_band = float(transition_band)
            stop_band_attenuation = float(stop_band_attenuation)
        except ValueError:
            messagebox.showerror("Error", "Please ensure all inputs are numeric.")
            return

        # Check for valid M and L values (both can't be zero)
        if M == 0 and L == 0:
            messagebox.showerror("Error", "Both M and L cannot be zero. Please enter valid values.")
            return

        if 'signals' not in globals() or signals is None:
            messagebox.showerror("Error", "No input signal loaded. Please load a signal first.")
            return

        input_signal = signals[0]['samples']
        global resampled_signal

        # FIR filter design
        _, coeffitents = FIR_filter_design('Low pass', fs, stop_band_attenuation, fc, transition_band)

        def remove_trailing_zeros(signal):
            # Find the last non-zero index and slice the signal
            last_non_zero_index = len(signal) - 1
            while last_non_zero_index >= 0 and signal[last_non_zero_index] == 0:
                last_non_zero_index -= 1
            return signal[:last_non_zero_index + 1]

        # Resampling logic
        if M == 0 and L != 0:  # Upsampling
            # Insert L-1 zeros between samples (upsampling)
            upsampled_signal = np.zeros(len(input_signal) * L)
            upsampled_signal[::L] = input_signal

            # Apply low-pass filter
            filtered_signal = np.convolve(upsampled_signal, coeffitents)
            filtered_signal = remove_trailing_zeros(filtered_signal)  # Remove trailing zeros
            resampled_signal = filtered_signal  # Return filtered signal after upsampling

        elif M != 0 and L == 0:  # Downsampling
            # Apply low-pass filter
            filtered_signal = np.convolve(input_signal, coeffitents)
            filtered_signal = remove_trailing_zeros(filtered_signal)  # Remove trailing zeros

            # Downsample by taking every M-th sample
            resampled_signal = filtered_signal[::M]  # Return downsampled signal

        elif M != 0 and L != 0:  # Fractional Resampling
            # Insert L-1 zeros between samples (upsampling)
            upsampled_signal = np.zeros(len(input_signal) * L)
            upsampled_signal[::L] = input_signal

            # Apply low-pass filter
            filtered_signal = np.convolve(upsampled_signal, coeffitents)
            filtered_signal = remove_trailing_zeros(filtered_signal)  # Remove trailing zeros

            # Downsample by taking every M-th sample
            resampled_signal = filtered_signal[::M]  # Return resampled signal after fractional resampling

        # Adjust the indices to start from -26
        adjusted_indices = list(range(-26, -26 + len(resampled_signal)))
        print("Indices   Samples:")
        for idx, sample in zip(adjusted_indices, resampled_signal):
            print(f"  {idx}        {sample}")

        # Handle comparison and signal saving based on L and M values
        if L == 0 and M == 2:
            Compare_Signals("Sampling_Down.txt", adjusted_indices, resampled_signal)
        elif L == 3 and M == 0:
            Compare_Signals("Sampling_Up.txt", adjusted_indices, resampled_signal)
        elif L == 3 and M == 2:
            Compare_Signals("Sampling_Up_Down.txt", adjusted_indices, resampled_signal)

        # Display success message
        messagebox.showinfo("Success", "Resampling completed successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
def open_resampling_popup(root):
    popup = tk.Toplevel(root)
    popup.title("Resampling Tool")
    popup.geometry("500x400")

    tk.Label(popup, text="Resampling Tool", font=(16)).pack(pady=10)

    # Frame for Inputs
    input_frame = tk.Frame(popup)
    input_frame.pack(pady=10, padx=20, fill="x")

    # Helper function to create labeled input fields
    def create_input(parent, label_text, var_type, row):
        tk.Label(parent, text=label_text, font=("Arial", 12)).grid(row=row, column=0, padx=10, pady=5,
                                                                                sticky="w")
        entry_var = var_type()
        entry = tk.Entry(parent, textvariable=entry_var, font=("Arial", 12), width=25)
        entry.grid(row=row, column=1, padx=10, pady=5)
        return entry_var

    fs_var = create_input(input_frame, "Sampling Frequency (FS):", tk.DoubleVar, 0,)
    fc_var = create_input(input_frame, "Cutoff Frequency (FC):", tk.DoubleVar, 1)
    transition_band_var = create_input(input_frame, "Transition Bandwidth:", tk.DoubleVar, 2)
    delta_s_var = create_input(input_frame, "Stopband Attenuation (dB):", tk.DoubleVar, 3)
    M_var = create_input(input_frame, "Decimation Factor (M):", tk.IntVar, 4)
    L_var = create_input(input_frame, "Interpolation Factor (L):", tk.IntVar, 5)

    # Buttons
    button_frame = tk.Frame(popup)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Load Signal", font=("Arial", 12), command=load_files).grid(row=0,
                                                                                                              column=0,
                                                                                                              padx=10)
    tk.Button(
        button_frame,
        text="Apply Resampling",
        font=("Arial", 12),
        command=lambda: resample_signal(M_var, L_var, fs_var, fc_var, transition_band_var, delta_s_var)
    ).grid(row=0, column=1, padx=10)
    tk.Button(button_frame, text="Cancel", font=("Arial", 12), command=popup.destroy).grid(row=1,
                                                                                                           column=1,
                                                                                                           padx=10)
