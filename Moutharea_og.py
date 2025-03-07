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

# Global variables to store clicked points
clicked_points = []


# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# File dialog for selecting the video file
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
if not video_path:
    print("No video selected. Exiting...")
    sys.exit()

input_filename = os.path.splitext(os.path.basename(video_path))[0]
save_directory = r'E:\Master\Cough project\Video\Output'  # Change this to the desired output directory

# Set the output paths dynamically based on input filename
save_video_path = os.path.join(save_directory, f"{input_filename}_output.mp4")
save_excel_path = os.path.join(save_directory, f"{input_filename}_mouth_area_data.xlsx")


'''
# File dialog for selecting the shape predictor model
model_path = filedialog.askopenfilename(title="Select Shape Predictor Model", filetypes=[("DAT Files", "*.dat"), ("All Files", "*.*")])
if not model_path:
    print("No model selected. Exiting...")
    sys.exit()
'''
'''
save_video_path = filedialog.asksaveasfilename(title="Select Video Save Path", defaultextension=".mp4", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
if not video_path:
    print("No path selected. Exiting...")
    sys.exit()
'''

# Load the selected video file and shape predictor model
cap = cv2.VideoCapture(video_path)  # Use the selected video file
#cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('E:\Master\Cough project\Code\Pythonproject\shape_predictor_68_face_landmarks.dat')  # Use the selected model

# Set up the video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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

# Load video and capture the first frame
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()

if ret:
    # Calibrate using clicks on the first frame
    pixels_per_cm = calibrate_with_clicks(first_frame)
else:
    print("Failed to load video.")
    sys.exit()



# Initialize an empty list to store mouth areas for each frame
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
        
        # Continue with your existing mouth area detection logic
        mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)], dtype=np.int32)
        area, hull = calculate_mouth_area(mouth_points)
        
        # Convert mouth area from pixels to cm² using the calibration
        real_mouth_area_cm2 = area / (pixels_per_cm ** 2)  # Since the area is in square pixels
        
        # Draw the mouth contour and display the scaled mouth area
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Pixel Mouth Area: {area:.2f} mm^2', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the real mouth area in the video frame
        cv2.putText(frame, f'Real Mouth Area: {real_mouth_area_cm2:.2f} cm^2', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
        # Store the mouth area for the current frame
        mouth_areas.append(area)
        frame_numbers.append(frame_count)    # Write the frame to the output video
        real_mouth_areas.append(real_mouth_area_cm2)
        
        
        
        
        
    out.write(frame)
    
    # Display the frame with the detected mouth area
    cv2.imshow('Mouth Area Frame by Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count += 1

# Plotting the graph of mouth area over time
plt.plot(frame_numbers, mouth_areas)
plt.xlabel('Frame Number')
plt.ylabel('Mouth Area')
plt.title('Mouth Area Over Time')
plt.grid(True)
plt.show()

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print or analyze the stored mouth areas
print("Mouth areas in video for each frame:", mouth_areas)
print("Mouth areas in real world for each frame:", real_mouth_areas)
print("Total frames", len(mouth_areas))

# Function to select save location
def select_save_path():
    """Helper function to open a dialog for saving the Excel file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")], title="Select File to Save")
    root.destroy()  # Destroy the root window after dialog is closed
    return save_path

# Normalizing the mouth areas based on the average of the first 10 frames and last 10 frames
if len(mouth_areas) >= 10:  # Ensure there are at least 10 frames
    # Calculate the average of the first 10 frames
    avg_first_10 = np.mean(mouth_areas[:10])
   
    # Calculate the average of the last 10 frames
    avg_last_10 = np.mean(mouth_areas[-10:])

    # Avoid division by zero if avg_first_10 and avg_last_10 are the same
    if avg_last_10 != avg_first_10:
        normalized_areas = [(area - avg_first_10) / (avg_last_10 - avg_first_10) for area in mouth_areas]
    else:
        normalized_areas = [0.5] * len(mouth_areas)  # If averages are the same, set all normalized values to 0.5
    
    
    
    # Create a pandas DataFrame with frame numbers and normalized areas
    frame_numbers = list(range(1, len(mouth_areas) + 1))  # Frame numbers start from 1
    data = {
        "Frame Number": frame_numbers,
        "Mouth Area": mouth_areas,
        "Real Area cm2": real_mouth_areas,
        "Normalized Mouth Area": normalized_areas,
    }
    df = pd.DataFrame(data)

    # Prompt the user to select the save location for the Excel file
    df.to_excel(save_excel_path, index=False)
    print(f"Data saved to {save_excel_path}")
else:
    print("Not enough frames to perform normalization based on the first and last 10 frames.")