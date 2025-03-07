# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:45:33 2024

@author: YZJef
"""
import cv2
import dlib
import numpy as np
import tkinter as tk
import sys
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

#Know real-world body height use factor of 0.13 to get head height
real_head_height_mm = float(input("Enter the body height (in m): "))*0.13*1000


# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# File dialog for selecting the video file
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
if not video_path:
    print("No video selected. Exiting...")
    sys.exit()

# File dialog for selecting the shape predictor model
model_path = filedialog.askopenfilename(title="Select Shape Predictor Model", filetypes=[("DAT Files", "*.dat"), ("All Files", "*.*")])
if not model_path:
    print("No model selected. Exiting...")
    sys.exit()

# Load the selected video file and shape predictor model
cap = cv2.VideoCapture(video_path)  # Use the selected video file
#cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)  # Use the selected model

# Set up the video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Function to calculate the mouth area
def calculate_mouth_area(points):
    hull = cv2.convexHull(points)
    return cv2.contourArea(hull), hull

def calculate_eye_distance(landmarks):
    # Right eye points (36 to 41)
    right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], dtype=np.int32)
    # Left eye points (42 to 47)
    left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], dtype=np.int32)
    
    # Calculate the center of each eye
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    
    # Calculate the Euclidean distance between the eye centers
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    
    return eye_distance, right_eye_center, left_eye_center

def calculate_head_height(landmarks):
    # Get the coordinates for the chin (landmark 8) and the nose bridge (landmark 27)
    chin_point = np.array([landmarks.part(8).x, landmarks.part(8).y])  # Chin
    nose_bridge_point = np.array([landmarks.part(27).x, landmarks.part(27).y])  # Nose bridge
    
    # Calculate the Euclidean distance between chin and nose bridge
    face_height = np.linalg.norm(chin_point - nose_bridge_point)
    
    # Estimate head height using a 1.5 multiplier to extrapolate to the top of the head
    head_height = face_height * 1.5  # Approximation for head height
    
    # Estimate the top of the head by extending the line from the nose bridge
    top_of_head = np.array([nose_bridge_point[0], nose_bridge_point[1] - (face_height * 0.5)])
    
    return head_height, chin_point, top_of_head

def scale_mouth_area(area, image_head_height):
    # Calculate the scaling factor
    scaling_factor = real_head_height_mm / image_head_height
    # Scale the area by the square of the scaling factor (since area is 2D)
    real_mouth_area = area * (scaling_factor ** 2)
    return real_mouth_area


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
        
        # Calculate head height and get the contour points (chin and estimated top of head)
        head_height, chin_point, top_of_head = calculate_head_height(landmarks)
        image_head_height = head_height
        
        # Calculate the eye distance
        eye_distance, right_eye_center, left_eye_center = calculate_eye_distance(landmarks)
       
        # Draw a line from the chin to the top of the head
        cv2.line(frame, tuple(chin_point), tuple(top_of_head.astype(int)), (255, 0, 0), 2)
        
        
        # Continue with your existing mouth area detection logic
        mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)], dtype=np.int32)
        area, hull = calculate_mouth_area(mouth_points)
        
       # Scale the mouth area to real-world units
        real_mouth_area = scale_mouth_area(area, image_head_height)
        
        # Draw the mouth contour and display the scaled mouth area
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        cv2.putText(frame, f'Real Mouth Area: {real_mouth_area:.2f} mm^2', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display head height on the frame
        cv2.putText(frame, f'Head Height: {image_head_height:.2f} px', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display real head height on the frame
        cv2.putText(frame, f'Real Head Height: {real_head_height_mm:.2f} mm', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Draw circles around the eye centers
        cv2.circle(frame, tuple(right_eye_center), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_eye_center), 3, (0, 255, 0), -1)

        # Draw the distance between the eyes
        cv2.line(frame, tuple(right_eye_center), tuple(left_eye_center), (255, 0, 0), 2)
        cv2.putText(frame, f'Eye Distance: {eye_distance:.2f} px', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow('Eye Distance', frame)
    
        # Store the mouth area for the current frame
        mouth_areas.append(area)
        frame_numbers.append(frame_count)
        real_mouth_areas.append(real_mouth_area)
    # Write the frame to the output video
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
    rw_avg_first_10 = np.mean(real_mouth_areas[:10])
    # Calculate the average of the last 10 frames
    avg_last_10 = np.mean(mouth_areas[-10:])
    rw_avg_last_10 = np.mean(real_mouth_areas[-10:])
    # Avoid division by zero if avg_first_10 and avg_last_10 are the same
    if avg_last_10 != avg_first_10:
        normalized_areas = [(area - avg_first_10) / (avg_last_10 - avg_first_10) for area in mouth_areas]
    else:
        normalized_areas = [0.5] * len(mouth_areas)  # If averages are the same, set all normalized values to 0.5
    
    if rw_avg_last_10 != rw_avg_first_10:
        rw_normalized_areas = [(real_mouth_area - rw_avg_first_10) / (rw_avg_last_10 - rw_avg_first_10) for real_mouth_area in real_mouth_areas]
    else:
        normalized_areas = [0.5] * len(real_mouth_areas)  # If averages are the same, set all normalized values to 0.5
    
    # Create a pandas DataFrame with frame numbers and normalized areas
    frame_numbers = list(range(1, len(mouth_areas) + 1))  # Frame numbers start from 1
    data = {
        "Frame Number": frame_numbers,
        "Mouth Area": mouth_areas,
        "Real World Mouth Area": real_mouth_areas,
        "Normalized Mouth Area": normalized_areas,
        "Normalized Real Mouth Area": rw_normalized_areas,
    }
    df = pd.DataFrame(data)

    # Prompt the user to select the save location for the Excel file
    output_filename = select_save_path()
    
    if output_filename:  # If a path is selected, save the DataFrame to the Excel file
        df.to_excel(output_filename, index=False)
        print(f"Data saved to {output_filename}")
    else:
        print("Save operation canceled.")
else:
    # If there are fewer than 10 frames, we cannot perform normalization
    print("Not enough frames to perform normalization based on the first and last 10 frames.")

