import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
from os.path import basename
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration parameters for data processing."""
    window_size: int = 7
    figure_size: Tuple[int, int] = (10, 6)
    x_ticks_step: int = 10
    x_ticks_max: int = 100

class DataProcessor:
    """Class to handle mouth area data processing and visualization."""
    
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
        self.max_values: List[float] = []
        
    @staticmethod
    def smooth_data_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply a moving average to smooth the data.
        
        Args:
            data: The input data to be smoothed
            window_size: The number of data points to average over
            
        Returns:
            The smoothed data array
            
        Raises:
            ValueError: If window_size is larger than data length
        """
        if window_size > len(data):
            raise ValueError(f"Window size ({window_size}) cannot be larger than data length ({len(data)})")
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    @staticmethod
    def shift_to_zero(data: np.ndarray) -> np.ndarray:
        """
        Shift the data to start at zero by subtracting the initial value.
        
        Args:
            data: The input data array
            
        Returns:
            The shifted data array starting at zero
        """
        return data - data[0]

    def process_video_data(self, data: np.ndarray, start_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process video data through the complete pipeline.
        
        Args:
            data: Raw mouth area data
            start_index: Starting frame index
            
        Returns:
            Tuple of (x_values, processed_data)
        """
        aligned_data = data[start_index:]
        smoothed_data = self.smooth_data_moving_average(aligned_data, self.config.window_size)
        shifted_data = self.shift_to_zero(smoothed_data)
        x_values = np.arange(len(shifted_data))
        return x_values, shifted_data

    def get_user_start_index(self, data: np.ndarray, video_column: str, file_path: str) -> int:
        """
        Get valid start index from user input.
        
        Args:
            data: The data array
            video_column: Name of the video column
            file_path: Path to the Excel file
            
        Returns:
            Valid start index
        """
        while True:
            try:
                start_index = int(input(f"Enter the frame number where the increase starts for {video_column} in {file_path}: "))
                if 0 <= start_index < len(data):
                    return start_index
                print(f"Start index must be between 0 and {len(data) - 1}. Try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def process_excel_file(self, file_path: str) -> None:
        """
        Process a single Excel file containing mouth area data.
        
        Args:
            file_path: Path to the Excel file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid
        """
        try:
            df = pd.read_excel(file_path)
            video_columns = [col for col in df.columns if 'Mouth Area' in col]
            
            if not video_columns:
                print(f"No mouth area data found in {file_path}.")
                return
                
            for video_column in video_columns:
                try:
                    mouth_area_data = df[video_column].dropna().values
                    if len(mouth_area_data) == 0:
                        print(f"No valid data found in column {video_column}")
                        continue
                        
                    start_index = self.get_user_start_index(mouth_area_data, video_column, file_path)
                    x_values, processed_data = self.process_video_data(mouth_area_data, start_index)
                    
                    plt.plot(x_values, processed_data, label=f"{video_column} ({basename(file_path)})")
                    
                    max_value = np.max(processed_data)
                    self.max_values.append(max_value)
                    print(f"Maximum value for {video_column}: {max_value:.2f}")
                    
                except Exception as e:
                    print(f"Error processing column {video_column}: {str(e)}")
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

    def plot_results(self) -> None:
        """Create and display the final plot."""
        plt.figure(figsize=self.config.figure_size)
        plt.xticks(range(0, self.config.x_ticks_max, self.config.x_ticks_step))
        plt.xlabel('Aligned Frame Number (Starting Point)')
        plt.ylabel('Mouth Area (Shifted & Smoothed)')
        plt.title('Aligned and Smoothed Mouth Area Over Time for Different Videos')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """Main function to run the data processing pipeline."""
    try:
        # Initialize tkinter and hide the main window
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        # Get Excel files from user
        excel_paths = filedialog.askopenfilenames(
            title="Select Excel Files with Mouth Area Data",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not excel_paths:
            print("No files selected. Exiting...")
            return

        # Initialize processor and process files
        processor = DataProcessor()
        total_files = len(excel_paths)
        
        print(f"\nProcessing {total_files} files...")
        for i, file_path in enumerate(excel_paths, 1):
            print(f"\nProcessing file {i}/{total_files}: {basename(file_path)}")
            processor.process_excel_file(file_path)

        # Display results
        if processor.max_values:
            average_max = np.mean(processor.max_values)
            print(f"\nAverage of maximum values: {average_max:.2f}")
            processor.plot_results()
        else:
            print("No valid data was processed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
