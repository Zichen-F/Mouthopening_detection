import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import os
import numpy as np

# Function to smooth data using a moving average
def smooth_data_moving_average(data, window_size):
    """
    Apply a moving average to smooth the data.
    
    Args:
        data (array-like): The input data to be smoothed.
        window_size (int): The number of data points to average over.
    
    Returns:
        np.ndarray: The smoothed data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    
    # Ask the user to select multiple Excel files
    file_paths = filedialog.askopenfilenames(title="Select Excel Files with Mouth Area Data", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    
    if not file_paths:
        print("No files selected. Exiting...")
        return

    for file_path in file_paths:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path)

        # Extract video names (column prefixes) by identifying columns with 'Mouth Area'
        video_columns = [col for col in df.columns if 'Mouth Area' in col]

        # Check if no video data is found
        if not video_columns:
            print(f"No mouth area data found in the file: {file_path}")
            continue
        
        # Get the file name without the extension for the plot title
        file_name = os.path.basename(file_path).split('.')[0]
        
        # Plot each video's data
        for video_column in video_columns:
            # Extract the mouth area data for the current video, drop NaN values
            mouth_area_data = df[video_column].dropna().values
            
            # Apply moving average smoothing to the data
            window_size = 7
            smoothed_data = smooth_data_moving_average(mouth_area_data, window_size=window_size)
            
            # Plot the smoothed data
            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_data, label=video_column)
            
            # Add labels and title to the plot
            plt.xlabel('Frame Number')
            plt.xticks(range(0, 150, 5))
            plt.ylabel('Mouth Area')
            plt.title(f'Mouth Area Over Time for {video_column} in {file_name} (Smoothed)')
            
            # Add legend to differentiate between videos
            plt.legend()

            # Show the grid and plot
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    main()
