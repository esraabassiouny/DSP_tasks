from tkinter import *
import tkinter as tk
from tkinter import ttk
import task1 as task1
import task3 as task3
import task4 as task4
import task5 as task5
import task2 as task2
import task6 as task6
import task7 as task7

def parse_input():
    amp = float(ampValue.get())
    theta = float(thetaValue.get())
    freq = float(freqValue.get())
    sampling_freq = float(samplingFreqValue.get())
    type = str(funct.get())
    return amp,theta ,freq,sampling_freq, type

# gui
root = tk.Tk()
root.title("Signal Generator")
root.geometry("500x400")
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)
browse_button = ttk.Button(frame, text="Open Signal File", command=lambda :task1.browse_file())
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
generateButton = ttk.Button(frame, text="Generate Signal", command=lambda :task1.generate_signal_and_compare(parse_input()))
generateButton.grid(row=7, column=0, columnspan=2, sticky="e")
task2_button = Button(text="Arithmetic Operations",command=lambda :task2.open_task2(root))
task2_button.pack()
task3_button = Button(root, text="Calculate Quantization", command=lambda :task3.open_task3(root))
task3_button.pack()
task4_button = Button(root, text="Fourier Transform", command=lambda :task4.open_task4(root))
task4_button.pack()
task5_button = Button(text="Time Domain",command=lambda :task5.open_task5(root))
task5_button.pack()
task6_button = Button(text="task 6",command=lambda : task6.open_task6(root))
task6_button.pack()
task71_button = Button(text="Filtering",command=lambda :task7.open_filtering_popup(root))
task71_button.pack()
task72_button = Button(text="Resampling",command=lambda : task7.open_resampling_popup(root))
task72_button.pack()
root.mainloop()