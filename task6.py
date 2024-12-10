from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt



def plot_signal(indices, signal, title="Signal"):
    """Plots a signal using provided indices and values."""
    plt.figure(figsize=(8, 6))
    plt.plot(indices, signal, label=title)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def smooth_signal():
    file1_path = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text files", "*.txt")])

    if not file1_path :
        messagebox.showerror("Input Error", "Please select both signal files.")
        return

    try:
        indices, signal = read_signal_with_indices_from_txt(file1_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the signals from the files: {e}")
        return

    window_size = simpledialog.askinteger("Input", "Enter the window size for smoothing:")

    if window_size is None:
        messagebox.showerror("Error", "Window size input canceled.")
        return
    if window_size <= 0:
        messagebox.showerror("Error", "Window size must be a positive integer.")
        return

    smoothed_signal = compute_moving_average(signal, window_size)
    smoothed_indices = indices[:len(smoothed_signal)]

    if "1" in file1_path:
        compare_signal("OutMovAvgTest1.txt",smoothed_signal,smoothed_indices)
    else:
        compare_signal("OutMovAvgTest2.txt",smoothed_signal,smoothed_indices)


def compute_moving_average(signal, window_size):
    if window_size < 2 or len(signal) <= window_size:
        raise ValueError("Window size must be greater than 1 and signal length must be greater than window size")
    smoothed_signal = []
    half_window = window_size // 2
    for n in range(half_window, len(signal) - half_window):
        avg = sum(signal[n - half_window:n + half_window + 1]) / window_size
        smoothed_signal.append(avg)

    return smoothed_signal


def compare_signal(file_path,smoothed_signal,smoothed_indices):
    if smoothed_signal is None or smoothed_indices is None:
        messagebox.showerror("Error", "No smoothed signal available. Please smooth a signal first.")
        return

    smoothed_signal = smoothed_signal.tolist() if isinstance(smoothed_signal, np.ndarray) else smoothed_signal

    SignalSamplesAreEquall(file_path, smoothed_indices, smoothed_signal)


def SignalSamplesAreEquall(file_name, indices, samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
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


def remove_dc_time():
    file1_path = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text files", "*.txt")])
    if not file1_path:
        messagebox.showerror("Input Error", "Please select both signal files.")
        return

    try:
        indices, signal = read_signal_with_indices_from_txt(file1_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the signals from the files: {e}")
        return
    mean_value = sum(signal) / len(signal)

    dc_removed_signal = [value - mean_value for value in signal]

    SignalSamplesAreEquall("DC_component_output.txt", indices, dc_removed_signal)

    plot_signal(indices, dc_removed_signal, title="DC Removed Signal (Time Domain)")
    print(f" Mean value: {mean_value}")
    return dc_removed_signal


def calculate_dft(signal):
    N = len(signal)
    amplitudes = np.zeros(N)
    phases = np.zeros(N)

    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            real_part += signal[n] * np.cos(angle)
            imag_part += -signal[n] * np.sin(angle)

        amplitudes[k] = np.sqrt(real_part ** 2 + imag_part ** 2)  # Magnitude of DFT
        phases[k] = np.arctan2(imag_part, real_part)  # Phase of DFT

    return amplitudes, phases, N


def calculate_idft(amplitudes, phases):
    N = len(amplitudes)
    signal_reconstructed = np.zeros(N)

    for n in range(N):
        real_part = 0
        imag_part = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            real_part += amplitudes[k] * np.cos(angle + phases[k])
            imag_part += amplitudes[k] * np.sin(angle + phases[k])

        signal_reconstructed[n] = real_part / N

    return signal_reconstructed


def remove_dc_freq():
    file1_path = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text files", "*.txt")])
    if not file1_path:
        messagebox.showerror("Input Error", "Please select both signal files.")
        return

    try:
        indices, signal = read_signal_with_indices_from_txt(file1_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the signals from the files: {e}")
        return

    amplitudes, phases, N = calculate_dft(signal)
    amplitudes[0] = 0
    dc_removed_signal = calculate_idft(amplitudes, phases)
    dc_removed_signal = dc_removed_signal.tolist()
    SignalSamplesAreEquall("DC_component_output.txt", indices, dc_removed_signal)
    plot_signal(indices, dc_removed_signal, title="DC Removed Signal (Frequency Domain)")


def convolve_signals():
    global signals
    file1_path = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text files", "*.txt")])
    file2_path = filedialog.askopenfilename(title="Select Second Signal File", filetypes=[("Text files", "*.txt")])

    if not file1_path or not file2_path:
        messagebox.showerror("Input Error", "Please select both signal files.")
        return
    try:
        indices1, samples1 = read_signal_with_indices_from_txt(file1_path)
        indices2, samples2 = read_signal_with_indices_from_txt(file2_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the signals from the files: {e}")
        return

    convolved_samples = []

    start_index = int(indices1[0] + indices2[0])
    end_index = int(indices1[-1] + indices2[-1])

    convolved_indices = list(range(start_index, end_index + 1))

    for k in range(len(convolved_indices)):
        conv_sum = 0
        for n in range(len(samples1)):
            j = k - n
            if 0 <= j < len(samples2):
                conv_sum += samples1[n] * samples2[j]
        convolved_samples.append(conv_sum)

    plot_signal(convolved_indices, convolved_samples, title="Convolved Signal")

    if "ConvTest" in globals():
        ConvTest(convolved_indices, convolved_samples)
    else:
        print("ConvTest function not defined. Skipping validation.")


def ConvTest(Your_indices, Your_samples):

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one")
            return
    print("Conv Test case passed successfully")


signals = []
last_correlation_indices = None
last_correlation_values = None


def normalized_cross_correlation(signal1, signal2):
    X1 = np.array(signal1)
    X2 = np.array(signal2)
    N = len(X1)
    X1_squared_sum = np.sum(X1 ** 2)
    X2_squared_sum = np.sum(X2 ** 2)
    normalization = np.sqrt(X1_squared_sum * X2_squared_sum)
    r12 = []
    for j in range(N):
        numerator = sum(X1[i] * X2[(i + j) % N] for i in range(N))
        r12.append(numerator / normalization)

    return np.array(r12)

def read_signal_with_indices_from_txt(file_path):
    indices = []
    samples = []
    try:
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        indices.append(int(parts[0]))
                        samples.append(float(parts[1]))
                    # else:
                    #     print(f"Skipping invalid line: {line}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")

    #print(f"Read {len(indices)} indices and {len(samples)} samples from {file_path}")  # Debugging output
    return indices, samples

def on_normalized_cross_correlation_click():
    file1_path = filedialog.askopenfilename(title="Select First Signal File", filetypes=[("Text files", "*.txt")])
    file2_path = filedialog.askopenfilename(title="Select Second Signal File", filetypes=[("Text files", "*.txt")])

    if not file1_path or not file2_path:
        messagebox.showerror("Input Error", "Please select both signal files.")
        return

    try:
        indices1, samples1 = read_signal_with_indices_from_txt(file1_path)
        indices2, samples2 = read_signal_with_indices_from_txt(file2_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the signals from the files: {e}")
        return
    output_signal = normalized_cross_correlation(samples1,samples2)
    print(output_signal)
    Compare_Signals("CorrOutput.txt",indices1,output_signal)


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
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
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one")
            return
    print("Correlation Test case passed successfully")


def open_task6(root):
    task6_window = Toplevel(root)
    task6_window.geometry("500x400")
    task6_window.title("Task 6")

    smooth_signal_button = Button(task6_window, text="Smooth Signal", command=smooth_signal)
    smooth_signal_button.pack()

    remove_dc_time_button = Button(task6_window, text="Remove DC (Time Domain)", command=remove_dc_time)
    remove_dc_time_button.pack()

    remove_dc_freq_button = Button(task6_window, text="Remove DC (Frequency Domain)", command=remove_dc_freq)
    remove_dc_freq_button.pack()

    convolve_button = Button(task6_window, text="convolve signals", command=convolve_signals)
    convolve_button.pack()

    corr_button = Button(task6_window, text="correlation signals", command=on_normalized_cross_correlation_click)
    corr_button.pack()


