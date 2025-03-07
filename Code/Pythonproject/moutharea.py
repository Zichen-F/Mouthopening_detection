# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:45:33 2024

@author: YZJef
"""

import cv2
import dlib
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

#Error handling
if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
    print("Shape predictor file not found. Please download it.")
    exit()

# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# File dialog for selecting the video file
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
if not video_path:
    print("No video selected. Exiting...")
    exit()

# File dialog for selecting the shape predictor model
model_path = filedialog.askopenfilename(title="Select Shape Predictor Model", filetypes=[("DAT Files", "*.dat"), ("All Files", "*.*")])
if not model_path:
    print("No model selected. Exiting...")
    exit()

cap = cv2.VideoCapture('video_path')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model_path')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


def calculate_mouth_area(points):
    hull = cv2.convexHull(points)
    return cv2.contourArea(hull), hull
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        landmarks = predictor(gray, face)
        mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)], dtype=np.int32)
        area, hull = calculate_mouth_area(mouth_points)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        cv2.putText(frame, f'Mouth Area: {area:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow('Mouth Area Frame by Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()