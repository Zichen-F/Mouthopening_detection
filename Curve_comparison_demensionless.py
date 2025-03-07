import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import os
import numpy as np

def normalize_curve(time_curve, data_curve):
    """
    Normalizes the given curve by:
    1. Scaling the time axis by dividing by peak time.
    2. Scaling the velocity data by its peak value.

    Parameters:
    - time_curve: NumPy array of time values for the curve
    - data_curve: NumPy array of velocity values for the curve

    Returns:
    - Normalized time values (t / t_peak)
    - Normalized velocity values (V / V_peak)
    """
    if len(data_curve) == 0 or np.all(data_curve == 0):
        print("Warning: Empty or all-zero data encountered. Skipping normalization.")
        return time_curve, data_curve  # No valid data to normalize

    time_curve = np.array(time_curve)
    data_curve = np.array(data_curve)

    # Find peak position (index of max value)
    peak_idx = np.argmax(data_curve)

    # Get peak velocity value and peak time
    peak_velocity = np.max(data_curve)
    peak_time = time_curve[peak_idx]

    # Avoid division by zero
    if peak_velocity == 0 or peak_time == 0:
        print("Warning: Peak velocity or peak time is zero. Returning unnormalized data.")
        return time_curve, data_curve

    norm_data = data_curve / peak_velocity  # Normalize velocity (V/V_peak)
    norm_time = time_curve / peak_time      # Normalize time (t/t_peak)

    return norm_time, norm_data

def main():
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select multiple Excel files
    file_paths = filedialog.askopenfilenames(
        title="Select Excel Files with Mouth Velocity Data",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )

    if not file_paths:
        print("No files selected. Exiting...")
        return

    for file_path in file_paths:
        try:
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path)

            # Identify columns for model, experimental, and constantA velocities
            mouth_velocity_model = [col for col in df.columns if 'Mouth_velocity_model' in col]
            mouth_velocity_exp = [col for col in df.columns if 'Mouth_velocity_exp' in col]
            mouth_velocity_constantA = [col for col in df.columns if 'Mouth_velocity_constantA' in col]

            # Check if required columns exist
            if not mouth_velocity_model or not mouth_velocity_exp or not mouth_velocity_constantA:
                print(f"Required columns missing in {file_path}. Skipping...")
                continue

            # Get the file name without extension for the plot title
            file_name = os.path.basename(file_path).split('.')[0]

            # Create the normalized plot
            plt.figure(figsize=(10, 5))

            for model_col, exp_col, constA_col in zip(mouth_velocity_model, mouth_velocity_exp, mouth_velocity_constantA):
                # Drop NaN values for clean data
                model_data = df[model_col].dropna()
                exp_data = df[exp_col].dropna()
                constA_data = df[constA_col].dropna()

                # Generate separate time axes
                time_model = model_data.index * 0.001  # Time in seconds
                time_exp = exp_data.index * 0.001  # Time in seconds
                time_constA = constA_data.index * 0.001  # Time in seconds

                # Normalize time and velocity data
                norm_time_model, norm_velocity_model = normalize_curve(time_model, model_data.values)
                norm_time_exp, norm_velocity_exp = normalize_curve(time_exp, exp_data.values)
                norm_time_constA, norm_velocity_constA = normalize_curve(time_constA, constA_data.values)

                # Plot normalized experimental data
                plt.plot(norm_time_exp, norm_velocity_exp, label="Experimental data from [9]", linestyle='--', marker='s', color='black')

                # Plot normalized model data
                plt.plot(norm_time_model, norm_velocity_model, label="Integrated model", linestyle='-', marker='o', color='blue')

                # Plot normalized constantA data
                plt.plot(norm_time_constA, norm_velocity_constA, label="Volume flow rate model from [6]", linestyle='-.', marker='^', color='red')

            # Labels, title, and legend
                plt.xlabel(r'Normalized Time ($t / t_{peak}$)')  # t_peak with subscript
                plt.xlim(0, 10)  # Since t/t_peak is normalized, we expect data mostly between 0 and ~2
                plt.ylabel(r'Normalized Mouth Velocity ($V / V_{peak}$)')  # V_peak with subscript
                plt.ylim(0, 1.2)  # Since all data is normalized, Y-axis is from 0 to 1.2
                plt.legend()
                plt.grid(True)
            # Show the normalized plot
            plt.show()

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    main()
