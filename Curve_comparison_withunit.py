import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import os
import numpy as np


def align_peak_time(time_exp, data_exp, time_curve, data_curve):
    """
    Aligns a given curve's peak time to the experimental peak time.
    
    Parameters:
    - time_exp: Time values for experimental data (reference curve)
    - data_exp: Experimental data values (reference curve)
    - time_curve: Time values for the curve being aligned
    - data_curve: Data values for the curve being aligned
    
    Returns:
    - Shifted time values for the aligned curve
    - Unchanged data values for the curve
    """
    if len(data_exp) == 0 or len(data_curve) == 0:
        return time_curve, data_curve  # No data to align

    # Find peak positions (index of max value)
    peak_exp_idx = np.argmax(data_exp)
    peak_curve_idx = np.argmax(data_curve)

    # Compute the time shift to align peaks
    peak_shift = time_exp[peak_exp_idx] - time_curve[peak_curve_idx]

    # Shift the curve's time data
    time_curve_shifted = time_curve + peak_shift

    return time_curve_shifted, data_curve




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

            # Create the original (aligned) plot
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

                # Align model and constantA data to match experimental peak time
                time_model_aligned, model_data_aligned = align_peak_time(time_exp, exp_data.values, time_model, model_data.values)
                time_constA_aligned, constA_data_aligned = align_peak_time(time_exp, exp_data.values, time_constA, constA_data.values)

                # Plot experimental data
                plt.plot(time_exp, exp_data, label="Experimental data from [9]", linestyle='--', marker='s', color='black')

                # Plot aligned model data
                plt.plot(time_model_aligned, model_data_aligned, label="Integrated model", linestyle='-', marker='o', color='blue')

                # Plot aligned constantA data
                plt.plot(time_constA_aligned, constA_data_aligned, label="Volume flow rate model from [6]", linestyle='-.', marker='^', color='red')

            # Labels, title, and legend
            plt.xlabel('Time (s)')
            plt.xlim(0, 0.6)
            plt.ylabel('Mouth Velocity (m/s)')
            plt.ylim(0, 25)
            plt.title('')
            plt.legend()
            plt.grid(True)

            # Show the original (aligned) plot
            plt.show()

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    main()
