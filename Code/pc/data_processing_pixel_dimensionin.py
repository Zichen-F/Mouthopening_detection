import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
from os.path import basename



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

# Function to shift data to start at zero by subtracting the initial value
def shift_to_zero(data):
    """
    Shift the data to start at zero by subtracting the initial value.
    Args:
        data (array-like): The input data.
    Returns:
        np.ndarray: The shifted data starting at zero.
    """
    initial_value = data[0]  # Get the first value
    return data - initial_value  # Subtract the first value from the entire data array

# Main function
def main():
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
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))
    max_values = []
    max_frame_numbers = []



    # Loop through each selected Excel file
    for file_path in excel_paths:
        df = pd.read_excel(file_path)

        # Extract video names (column prefixes) by identifying columns with 'Mouth Area'
        video_columns = [col for col in df.columns if 'Mouth Area' in col]

        # Check if no video data is found
        if not video_columns:
            print(f"No mouth area data found in {file_path}.")
            continue

        # Loop over each video's data
        for video_column in video_columns:
            
            # Extract the mouth area data for the current video, drop NaN values
            mouth_area_data = df[video_column].dropna().values

            # Allow the user to input both start and end indices
            while True:
                try:
                    start_index = int(input(f"Enter the start frame number for {video_column} in {file_path}: "))
                    end_index = int(input(f"Enter the end frame number for {video_column} in {file_path} (Enter -1 to use all remaining data): "))
            
                    if 0 <= start_index < len(mouth_area_data):
                        if end_index == -1:  # Allow full length processing
                            end_index = len(mouth_area_data)
                        if start_index < end_index <= len(mouth_area_data):
                            break  # Valid range
                        else:
                            print(f"End index must be greater than start index and within range (0 to {len(mouth_area_data)}). Try again.")
                    else:
                        print(f"Start index must be between 0 and {len(mouth_area_data) - 1}. Try again.")
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            
            # Apply moving average smoothing to the data
            window_size = 7
            smoothed_data = smooth_data_moving_average(mouth_area_data, window_size=window_size)
            
            # Slice the raw data from the user-defined start point
            aligned_raw_data = smoothed_data[start_index:]

            # Shift the data to start at zero
            shifted_data = shift_to_zero(aligned_raw_data)
            # Ignore all final values below zero
            valid_indices = shifted_data >= 0
            shifted_data = shifted_data[valid_indices]
            x_values = np.arange(len(shifted_data))/40  # Update x-values to match filtered data
            # Ensure there's still valid data after filtering
            if len(shifted_data) == 0:
                print(f"All final values are below zero for {video_column} in {file_path}. Skipping...")
                continue
        
        
            # Plot the shifted data
            plt.plot(x_values, shifted_data, label=f"{video_column} ({basename(file_path)})")
            
            # Find and store the maximum value and its frame number after alignment and smoothing
            max_value = np.max(shifted_data)
            max_values.append(max_value)
            max_frame_number = np.argmax(shifted_data)  # Adjust by the start index
            max_frame_numbers.append(max_frame_number)
            print(f"Maximum value for {video_column} in {file_path} after smoothing and alignment: {max_value}")
            print(f"Frame number of maximum value: {max_frame_number}")

    # Calculate and print the average of the max values
    average_max_value = np.mean(max_values)
    print(f"Average of maximum values after smoothing: {average_max_value}")

    # Calculate and print the average frame number of maximum mouth opening
    if max_frame_numbers:
        average_frame_number = np.mean(max_frame_numbers)
        print(f"Average frame number where maximum mouth opening happens: {average_frame_number}")
    else:
        print("No valid data to calculate average frame number.")

    # Add labels and title to the plot
    # Set x-axis ticks and limits dynamically
    plt.xticks(np.arange(0, 2, step=0.1))  # Tick step of 0.1
    plt.xlim(0, 2)  # Dynamically adjust x-axis range
    plt.ylim(0, 5)
    plt.xlabel('Time (s)')
    plt.ylabel('Mouth Pixel Area')

    # Add legend to differentiate between videos
    # Show the grid and plot
    plt.grid(True)
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
