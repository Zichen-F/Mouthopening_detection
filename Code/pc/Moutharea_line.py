# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:04:49 2024

@author: YZJef
"""
import os
import cv2
import dlib
import numpy as np
import tkinter as tk
import sys
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

# Global variables to store clicked points and mouth data
clicked_points = []



# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# Function to calculate the mouth area
def calculate_mouth_area(points):
    hull = cv2.convexHull(points)
    return cv2.contourArea(hull), hull

# Mouse callback function to capture the clicked points
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point when left button is clicked
        clicked_points.append((x, y))
        cv2.circle(params['frame'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration Frame", params['frame'])

# Function to calibrate the scale using user-selected points
def calibrate_with_clicks(frame):
    global clicked_points
    clicked_points = []  # Reset any previous points

    # Display the first frame and let the user click two points
    cv2.imshow("Calibration Frame", frame)
    cv2.setMouseCallback("Calibration Frame", click_event, param={'frame': frame})
    print("Click two points on the frame to define the known distance.")

    # Wait until two points are clicked
    while len(clicked_points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calculate the distance between the two points in pixels
    x1, y1 = clicked_points[0]
    x2, y2 = clicked_points[1]
    pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Ask the user for the real-world distance between the clicked points
    real_distance = float(input("Enter the real-world distance between the points (in cm): "))

    # Calculate pixels per cm
    pixels_per_cm = pixel_distance / real_distance
    print(f"Calibration successful! {pixels_per_cm:.2f} pixels per cm.")

    return pixels_per_cm

# Function to process a video and return its data
def process_video(video_path):
    # Load the video file and shape predictor model
    cap = cv2.VideoCapture(video_path)  # Use the selected video file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('E:\Master\Cough project\Code\Pythonproject\shape_predictor_68_face_landmarks.dat')  # Use the selected model
    
    # Load video and capture the first frame
    ret, first_frame = cap.read()
    
    if not ret:
        print("Failed to load video.")
        sys.exit()
    
    # Calibrate using clicks on the first frame
    pixels_per_cm = calibrate_with_clicks(first_frame)
    
    
 
    
    # Initialize data lists
    mouth_areas = []
    real_mouth_areas = []
    frame_numbers = []
    
    # Process the video frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Continue with mouth area detection
            mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)], dtype=np.int32)
            area, hull = calculate_mouth_area(mouth_points)
            
            # Convert mouth area from pixels to cm² using calibration
            real_mouth_area_cm2 = area / (pixels_per_cm ** 2)  # Area in square pixels

            # Draw the mouth contour and display the scaled mouth area
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
 
            cv2.putText(frame, f'Pixel Mouth Area: {area:.2f} mm^2', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
            # Display the real mouth area in the video frame
            cv2.putText(frame, f'Real Mouth Area: {real_mouth_area_cm2:.2f} cm^2', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Store the mouth area for the current frame
            mouth_areas.append(area)
            real_mouth_areas.append(real_mouth_area_cm2)
            frame_numbers.append(frame_count)


        frame_count += 1

    # Release video resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Normalizing the mouth areas based on the average of the first 10 frames and last 10 frames
    if len(mouth_areas) >= 10:
        avg_first_10 = np.mean(mouth_areas[:10])
        avg_last_10 = np.mean(mouth_areas[-10:])
        
        # Avoid division by zero if averages are the same
        if avg_last_10 != avg_first_10:
            normalized_areas = [(area - avg_first_10) / (avg_last_10 - avg_first_10) for area in mouth_areas]
        else:
            normalized_areas = [0.5] * len(mouth_areas)
    else:
        normalized_areas = [0.5] * len(mouth_areas)

    # Return the processed data as a dictionary
    return {
        "Frame Number": frame_numbers,
        "Mouth Area": mouth_areas,
        "Real Area cm²": real_mouth_areas,
        "Normalized Mouth Area": normalized_areas
    }

# Main function to run the program and handle multiple videos
def main():
    # Initialize an empty DataFrame to store all videos' data
    all_videos_data = pd.DataFrame()
    
    # Counter to track the video number
    video_count = 1

    # Loop to keep processing videos
    while True:
        # File dialog for selecting the video file
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
        if not video_path:
            print("No video selected. Exiting...")
            break
        
        # Process the selected video
        video_data = process_video(video_path)

        # Convert video data to a DataFrame
        df = pd.DataFrame(video_data)

        # Add columns with the video number
        video_prefix = f"Video {video_count}"
        df.columns = [f"{video_prefix} - {col}" for col in df.columns]

        # Combine the new video data with the existing data in all_videos_data
        all_videos_data = pd.concat([all_videos_data, df], axis=1)
        
        # Increment video count
        video_count += 1

        # Ask the user if they want to process another video
        cont = input("Do you want to process another video? (y/n): ").strip().lower()
        if cont != 'y':
            break

    # Ask for the save location for the Excel file
    save_excel_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")], title="Save Combined Excel File")
    if save_excel_path:
        # Save the combined data to an Excel file
        all_videos_data.to_excel(save_excel_path, index=False)
        print(f"Data saved to {save_excel_path}")
    else:
        print("No save location selected. Exiting without saving data.")

# Run the main function
if __name__ == "__main__":
    main()
