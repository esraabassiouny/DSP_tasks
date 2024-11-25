from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import shared as sh

def SignalSamples_Dct(file_name, indices, samples):
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


def DCT(signal, m):
    N = len(signal)
    dct_coefficients = np.zeros(N)  # Initialize DCT coefficient array

    for k in range(N):  # k ranges from 1 to N
        summation = 0
        for n in range(N):  # n ranges from 1 to N
            summation += signal[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
        dct_coefficients[k] = np.sqrt(2 / N) * summation

    # Save only the first m coefficients if specified
    if m is not None and m < N:
        dct_coefficients = dct_coefficients[:m]

    return dct_coefficients


def on_dct_button_click():
    # Let user select a signal file
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        messagebox.showerror("File Error", "No file selected.")
        return

    # Read the signal from the file
    signal = sh.read_signal_from_file(file_path)  # Assuming this function is already implemented
    if signal is None:
        messagebox.showerror("Error", "Could not read signal from the file.")
        return

    # Ask user for the number of coefficients (m) to save
    m = simpledialog.askinteger("Input", "Enter the number of DCT coefficients to save:")
    if m is None or m <= 0:
        messagebox.showerror("Input Error", "Invalid number of coefficients.")
        return

    # Compute the DCT
    dct_coefficients = DCT(signal, m)

    # Save the first m coefficients to a new file
    output_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if not output_file:
        messagebox.showerror("File Error", "No output file selected.")
        return

    with open(output_file, "w") as f:
        f.write("0\n")  # Add header or meta information if needed
        f.write("1\n")
        f.write(f"{len(dct_coefficients)}\n")  # Write the number of coefficients (m)
        for index, coeff in enumerate(dct_coefficients):
            f.write(f"{index} {coeff:.6f}\n")

    # Display the result
    print("DCT Coefficients:")
    for index, coeff in enumerate(dct_coefficients):
        print(f"{index}: {coeff:.6f}")

    messagebox.showinfo("Success", f"DCT coefficients saved to {output_file}")

    # Compare the output file with the original signal
    SignalSamples_Dct("DCT_output.txt", range(len(dct_coefficients)), dct_coefficients)  # Compare output file with the DCT coefficients


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

def fourier_transform(freq):
    file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    freq = int (freq)
    if fourier_choice.get() == 1:
        signal = sh.read_signal_from_file(file)
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
        if SignalComapreAmplitude(amp,output_amp)& SignalComaprePhaseShift(phase_shift,output_phase):
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
        output_freq = sh.read_signal_from_file("Output_Signal_IDFT.txt")
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


def open_task4(root):
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
    load_button = Button(task4window, text="Upload and Compute DCT", command=on_dct_button_click)
    load_button.pack()

    Label(task4window, text="Enter Sampling Frequency:").pack()
    sampling_freq = Entry(task4window, width=20)
    sampling_freq.pack()

    # Button to start quantization process
    FT_button = Button(task4window, text="Calculate Signal", command=lambda :fourier_transform(sampling_freq.get()))
    FT_button.pack()