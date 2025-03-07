import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
from os.path import basename



# Function to smooth data using a moving average
def smooth_data_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to shift data to start at zero
def shift_to_zero(data):
    return data - data[0]

# Function to calculate the theoretical function f(t)
def f(t, t_max, C, a, b):
    x = t / t_max
    return C * np.exp(-a * x) * (1 - np.exp(-x)) ** (b - 1)

def main():
    # Initialize tkinter for file selection
    root = tk.Tk()
    root.withdraw()

    # Select Excel files
    excel_paths = filedialog.askopenfilenames(title="Select Excel Files", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    if not excel_paths:
        print("No files selected. Exiting...")
        return

    # Initialize variables for plotting
    x_values_combined = []
    shifted_data_combined = []

    # Process each Excel file
    for file_path in excel_paths:
        df = pd.read_excel(file_path)

        # Identify columns with 'Mouth Area'
        video_columns = [col for col in df.columns if 'Mouth Area' in col]
        if not video_columns:
            print(f"No mouth area data found in {file_path}.")
            continue

        for video_column in video_columns:
            mouth_area_data = df[video_column].dropna().values

            # Prompt for start index
            while True:
                try:
                    start_index = int(input(f"Enter the start index for '{video_column}' in {basename(file_path)} (0 to {len(mouth_area_data) - 1}): "))
                    if 0 <= start_index < len(mouth_area_data):
                        break
                except ValueError:
                    print("Invalid input. Enter an integer.")

            smoothed_data = smooth_data_moving_average(mouth_area_data, window_size=7)
            aligned_data = smoothed_data[start_index:]
            shifted_data = shift_to_zero(aligned_data)
            x_values = np.arange(len(shifted_data))
            x_values_combined.extend(x_values)
            shifted_data_combined.extend(shifted_data)
            # Find and display max shifted data and corresponding x-value
            max_shifted_value = np.max(shifted_data)
            max_shifted_index = np.argmax(shifted_data)
            print(f"File: {basename(file_path)}, Column: {video_column}")
            print(f"Max shifted data: {max_shifted_value}, Corresponding x: {max_shifted_index}, Time (s): {max_shifted_index / 40:.2f}s")
    # Constants for theoretical function
    a = 2.15
    b = 4.51
    t_max = 0.2
    a_max = 1479.93
    time_sec = np.array(x_values_combined) / 40
    C = 1 / (np.exp(-a * 1) * (1 - np.exp(-1)) ** (b - 1))

    # Generate theoretical data
    t_values = np.linspace(0, 2, 600)
    a_values = f(t_values, t_max, C, a, b) * a_max
    
    scaling_factor = 0.0029  # Change this as needed
    shifted_data_combined = np.array(shifted_data_combined) * scaling_factor
    a_values = a_values * scaling_factor

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, a_values, label="Fitted A(t)", color="blue")
    plt.plot(time_sec, shifted_data_combined, label="Mouth opening measured", color="orange")
    plt.xlim(0,2)
    plt.ylim(0,4.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Area (cm)")
    plt.title("")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
