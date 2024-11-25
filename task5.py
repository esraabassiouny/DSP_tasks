from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt


def derivative_first_signal():
    InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                   19.0,
                   20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                   37.0,
                   38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
                   53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,
                   69.0,
                   70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
                   86.0,
                   87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    expectedOutput_first = [1] * (len(InputSignal) - 1)

    # Compute the first derivative
    FirstDrev = [InputSignal[i] - InputSignal[i - 1] for i in range(1, len(InputSignal))]

    # Validation
    if len(FirstDrev) != len(expectedOutput_first) or any(
            abs(FirstDrev[i] - expectedOutput_first[i]) >= 0.01 for i in range(len(FirstDrev))):
        messagebox.showerror("Error", "First Derivative Test case failed!")
    else:
        messagebox.showinfo("Success", "First Derivative Test case passed successfully!")


def derivative_second_signal():
    InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                   18.0,
                   19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0,
                   35.0,
                   36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0,
                   52.0,
                   53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,
                   69.0,
                   70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
                   86.0,
                   87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    expectedOutput_second = [0] * (len(InputSignal) - 2)

    # Compute the second derivative
    SecondDrev = [InputSignal[i + 1] - 2 * InputSignal[i] + InputSignal[i - 1] for i in
                  range(1, len(InputSignal) - 1)]

    # Validation
    if len(SecondDrev) != len(expectedOutput_second) or any(
            abs(SecondDrev[i] - expectedOutput_second[i]) >= 0.01 for i in range(len(SecondDrev))):
        messagebox.showerror("Error", "Second Derivative Test case failed!")
    else:
        messagebox.showinfo("Success", "Second Derivative Test case passed successfully!")


signals = []


def shift(txt):
    input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
    xShFo = []
    x_indices = []
    ySignal = []
    with open(input_file, "r") as f:
        for i in range(3):
            next(f)
        for line in f:
            parts = line.strip().split()
            x_indices.append(float(parts[0]))
            ySignal.append(float(parts[1]))

    const = int(txt)
    for i in range(len(x_indices)):
        xShFo.append(x_indices[i] + const)
    print(xShFo)
    figure, shift = plt.subplots(2, 1, figsize=(6, 8))
    shift[0].plot(x_indices, ySignal)
    shift[0].set_title("Original signal")
    shift[1].plot(xShFo, ySignal)
    shift[1].set_title("Shifted signal")
    plt.show()


def shift_and_fold_signal(txt):
    input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
    xShFo = []
    x_indices = []
    ySignal = []
    yFolded = []
    with open(input_file, "r") as f:
        for i in range(3):
            next(f)
        for line in f:
            parts = line.strip().split()
            x_indices.append(float(parts[0]))
            ySignal.append(float(parts[1]))

    for i in range(len(ySignal) - 1, -1, -1):
        yFolded.append(ySignal[i])
    const = int(txt)
    for i in range(len(x_indices)):
        xShFo.append(x_indices[i] + const)
    figure, shift = plt.subplots(2, 1, figsize=(6, 8))
    shift[0].plot(x_indices, ySignal)
    shift[0].set_title("Original signal")
    shift[1].plot(xShFo, yFolded)
    shift[1].set_title("Shifted signal")
    plt.show()
    if const == 500:
        Shift_Fold_Signal("Output_ShifFoldedby500.txt", xShFo, yFolded)
    else:
        Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", xShFo, yFolded)


def upload_file_and_fold():
    input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
    if not input_file:
        print("No input file selected.")
        return

    xSignal = []
    ySignal = []
    yFolded = []
    expected_t = []
    expected_signal = []
    with open(input_file, "r") as f:
        for i in range(3):
            next(f)

        for line in f:
            parts = line.strip().split()
            xSignal.append(float(parts[0]))
            ySignal.append(float(parts[1]))

    for i in range(len(ySignal) - 1, -1, -1):
        yFolded.append(ySignal[i])

    output_file = "Output_fold.txt"

    with open(output_file, "r") as f:
        for i in range(3):
            next(f)

        for line in f:
            parts = line.strip().split()
            expected_t.append(float(parts[0]))
            expected_signal.append(float(parts[1]))

    # Plot the signals for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(xSignal, ySignal, label="Original Signal")
    plt.plot(xSignal, yFolded, label="Folded Signal", linestyle="--")
    plt.plot(expected_t, expected_signal, label="Expected Folded Signal", linestyle=":")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal Folding and Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    Shift_Fold_Signal("Output_fold.txt", xSignal, yFolded)


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
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
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    print("Shift_Fold_Signal Test case passed successfully")


def open_task5(root):
    task5_window = Toplevel(root)
    task5_window.geometry("500x400")
    task5_window.title("Task 5 - Time Domain")
    first_signal = Button(task5_window, text="first_signal", command=derivative_first_signal)
    first_signal.pack()
    second_signal = Button(task5_window, text="second_signal", command=derivative_second_signal)
    second_signal.pack()
    folding_button = Button(task5_window, text="Fold Signal", command=upload_file_and_fold)
    folding_button.pack()
    lf = Label(task5_window, text="Constant")
    txt = Text(task5_window, width="50", height="2")
    lf.pack()
    txt.pack()
    shifting_and_fold = Button(task5_window, text="Shift and Fold Signal",
                               command=lambda: shift_and_fold_signal(txt.get(1.0, 'end')))
    shifting_and_fold.pack()
    shifting = Button(task5_window, text="Shift", command=lambda: shift(txt.get(1.0, 'end')))
    shifting.pack()
