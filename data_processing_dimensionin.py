import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
from os.path import basename

# Function to smooth data using a moving average
def smooth_data_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to shift data to start at zero by subtracting the initial value
def shift_to_zero(data):
    initial_value = data[0]
    return data - initial_value

# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# Bring the file dialog to the front
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)

# Ask the user to select multiple Excel files
excel_paths = filedialog.askopenfilenames(title="Select Excel Files with Mouth Area Data", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
if not excel_paths:
    print("No files selected. Exiting...")
    exit()

# Set up the plot
plt.figure(figsize=(10, 6))
max_values = []
max_frame_numbers = []

# Loop through each selected Excel file
for file_path in excel_paths:
    df = pd.read_excel(file_path)

    # Extract video names (column prefixes) by identifying columns with 'Real Area'
    video_columns = [col for col in df.columns if 'Real Area' in col]

    if not video_columns:
        print(f"No mouth area data found in {file_path}.")
        continue

    for video_column in video_columns:
        mouth_area_data = df[video_column].dropna().values

        # Allow user to input start and end indices
        while True:
            try:
                start_index = int(input(f"Enter the start frame for {video_column} in {file_path}: "))
                end_index = int(input(f"Enter the end frame for {video_column} (or -1 for full data): "))

                if 0 <= start_index < len(mouth_area_data):
                    if end_index == -1:
                        end_index = len(mouth_area_data)
                    if start_index < end_index <= len(mouth_area_data):
                        break
                    else:
                        print(f"End index must be within range (0 to {len(mouth_area_data)}). Try again.")
                else:
                    print(f"Start index must be between 0 and {len(mouth_area_data) - 1}. Try again.")
            except ValueError:
                print("Invalid input. Enter an integer.")

        # Apply moving average smoothing
        window_size = 7
        smoothed_data = smooth_data_moving_average(mouth_area_data, window_size)

        # Slice and shift data
        aligned_raw_data = smoothed_data[start_index:]
        shifted_data = shift_to_zero(aligned_raw_data)

        # Ignore final values below zero
        valid_indices = shifted_data >= 0
        shifted_data = shifted_data[valid_indices]
        x_values = np.arange(len(shifted_data)) / 40

        if len(shifted_data) == 0:
            print(f"All final values are below zero for {video_column} in {file_path}. Skipping...")
            continue

        # Plot the shifted data
        plt.plot(x_values, shifted_data, label=f"{video_column} ({basename(file_path)})")

        # Find and store the maximum value
        max_value = np.max(shifted_data)
        max_values.append(max_value)
        max_frame_number = np.argmax(shifted_data)
        max_frame_numbers.append(max_frame_number)

        print(f"Maximum value for {video_column} in {file_path}: {max_value}")
        print(f"Frame number of maximum value: {max_frame_number}")

# Calculate statistics
if max_values:
    print(f"Max mouth opening: {np.max(max_values)}")
    print(f"Min mouth opening: {np.min(max_values)}")
    print(f"Average mouth opening: {np.mean(max_values):.2f}")
    print(f"Standard deviation: {np.std(max_values):.2f}")

if max_frame_numbers:
    print(f"Max time of opening: {np.max(max_frame_numbers) / 40:.2f} s")
    print(f"Min time of opening: {np.min(max_frame_numbers) / 40:.2f} s")
    print(f"Average time of opening: {np.mean(max_frame_numbers) / 40:.2f} s")
    print(f"Standard deviation of time: {np.std(max_frame_numbers) / 40:.2f} s")

# Configure the plot
plt.xticks(np.arange(0, 2.1, step=0.1))
plt.yticks(np.arange(0, 5.1, step=0.5))
plt.xlim(0, 2)
plt.ylim(0, 5)
plt.xlabel('Time (s)')
plt.ylabel('Mouth Opening Area (cm²)')
plt.grid(True)
plt.legend()
plt.show()
