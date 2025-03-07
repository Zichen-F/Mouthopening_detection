import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
from os.path import basename, splitext

# Function to smooth data using a moving average
def smooth_data_moving_average(data, window_size):
    """
    Apply a moving average to smooth the data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to shift data to start at zero by subtracting the initial value
def shift_to_zero(data):
    """
    Shift the data to start at zero by subtracting the initial value.
    """
    initial_value = data[0]
    return data - initial_value

def main():
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    
    # Bring the file dialog to the front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # Ask the user to select multiple Excel files
    excel_paths = filedialog.askopenfilenames(
        title="Select Excel Files with Mouth Area Data", 
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not excel_paths:
        print("No files selected. Exiting...")
        return

    # Ask user for a save directory only ONCE
    save_path = filedialog.askdirectory(title="Select a Directory to Save Processed Files")
    if not save_path:
        print("No save directory selected. Exiting...")
        return

    # Create a single figure
    fig_all, ax_all = plt.subplots(figsize=(10, 6))

    # Store max values and Tmax values for averaging
    max_values = []
    t_max_values = []

    for file_path in excel_paths:
        df = pd.read_excel(file_path)
        file_name = basename(file_path)
        file_stem, _ = splitext(file_name)  # Get file name without extension

        # Identify columns containing 'Mouth Area'
        video_columns = [col for col in df.columns if 'Mouth Area' in col]

        if not video_columns:
            print(f"No mouth area data found in {file_path}. Skipping...")
            continue

        # Dictionary to store processed data for the current file
        processed_data_dict = {}

        for video_column in video_columns:
            mouth_area_data = df[video_column].dropna().values

            # User input for start and end indices
            while True:
                try:
                    start_index = int(input(f"Enter the start frame number for {video_column} in {file_path}: "))
                    end_index = int(input(f"Enter the end frame number for {video_column} in {file_path} (Enter -1 to use all remaining data): "))

                    if 0 <= start_index < len(mouth_area_data):
                        if end_index == -1:  # Allow user to process till the last frame
                            end_index = len(mouth_area_data)
                        if start_index < end_index <= len(mouth_area_data):
                            break
                        else:
                            print(f"End index must be greater than the start index and within range (0 to {len(mouth_area_data)}). Try again.")
                    else:
                        print(f"Start index must be between 0 and {len(mouth_area_data) - 1}. Try again.")
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            
            # Slice data from the start to the end index
            aligned_raw_data = mouth_area_data[start_index:end_index]

            # Smooth the data
            window_size = 7
            smoothed_data = smooth_data_moving_average(aligned_raw_data, window_size=window_size)

            # Shift data to start at zero
            shifted_data = shift_to_zero(smoothed_data)

            # Find max value and Tmax
            max_value = np.max(shifted_data)
            t_max = np.argmax(shifted_data)

            if max_value == 0:
                print(f"Maximum value for {video_column} in {file_path} is zero. Skipping normalization.")
                continue

            # Store max values for averaging
            max_values.append(max_value)
            t_max_values.append(t_max)

            # Normalize the data
            normalized_data = shifted_data / max_value
            
            # Compute dimensionless time
            t_values = np.arange(len(normalized_data))
            t_tmax = t_values / t_max if t_max != 0 else t_values
            
            # Ignore all final values below zero
            valid_indices = normalized_data >= 0
            t_tmax = t_tmax[valid_indices]
            normalized_data = normalized_data[valid_indices]
            
            # Ensure there's still data left after filtering
            if len(normalized_data) == 0:
                print(f"All final values are below zero for {video_column} in {file_path}. Skipping...")
                continue

            # Plot all data
            ax_all.plot(t_tmax, normalized_data, label=f"{video_column} ({file_name})")

            # Store processed data in dictionary
            data_df = pd.DataFrame({
                't_tmax': t_tmax,
                'normalized_data': normalized_data
            })
            processed_data_dict[video_column] = data_df

            print(f"Processed {video_column} in {file_path}: Max Value = {max_value}, Tmax = {t_max}")

        # Save the processed data for the current file
        output_file_path = f"{save_path}/{file_stem}_processed.xlsx"
        with pd.ExcelWriter(output_file_path) as writer:
            for video_column, processed_data in processed_data_dict.items():
                processed_data.to_excel(writer, sheet_name=video_column)

        print(f"Processed data for {file_name} saved to {output_file_path}")

    # Calculate and print average max and t_max
    if max_values and t_max_values:
        avg_max_value = np.mean(max_values)
        avg_t_max = np.mean(t_max_values)
        print(f"\n### Final Results Across All Files ###")
        print(f"Average Max Value: {avg_max_value:.2f}")
        print(f"Average Tmax: {avg_t_max:.2f}")
    else:
        print("\nNo valid max values found for averaging.")

    # Set x and y limits for the final plot
    ax_all.set_xlim(0, 8)  # Set x-axis range from 0 to 4
    ax_all.set_ylim(0, 1.2)  # Set y-axis range from 0 to 1.2

    # Set labels, title, and legend for the combined plot
    ax_all.set_xlabel(r'$\tau$')
    ax_all.set_ylabel(r'$\tilde{A}$')
    ax_all.set_title('')
    ax_all.grid(True)

    # Display the figure
    plt.show()

if __name__ == "__main__":
    main()
