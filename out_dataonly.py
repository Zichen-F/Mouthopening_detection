
import os
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Global variables for clicked points
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
        clicked_points.append((x, y))
        cv2.circle(params['frame'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration Frame", params['frame'])

# Function to calibrate the scale using user-selected points
def calibrate_with_clicks(frame):
    global clicked_points
    clicked_points = []  # Reset points

    # Display the frame and allow user to click
    cv2.imshow("Calibration Frame", frame)
    cv2.setMouseCallback("Calibration Frame", click_event, param={'frame': frame})
    print("Click two points on the frame to define the known distance.")

    # Wait until two points are clicked
    while len(clicked_points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calculate pixel distance
    x1, y1 = clicked_points[0]
    x2, y2 = clicked_points[1]
    pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Get real-world distance
    real_distance = float(input("Enter the real-world distance between the points (in cm): "))

    # Calculate pixels per cm
    pixels_per_cm = pixel_distance / real_distance
    print(f"Calibration successful! {pixels_per_cm:.2f} pixels per cm.")
    return pixels_per_cm

# Function to process frame images
def process_frames(frame_folder):
    # Load shape predictor model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:\\Master\\Cough project\\Code\\Pythonproject\\shape_predictor_68_face_landmarks.dat')

    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp', '.tiff'))])

    if not frame_files:
        print("No frame images found in the selected folder.")
        return

    # Load the first frame for calibration
    first_frame_path = os.path.join(frame_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print("Failed to load the first frame.")
        return

    # Calibrate using the first frame
    pixels_per_cm = calibrate_with_clicks(first_frame)

    # Initialize data lists
    mouth_areas = []
    real_mouth_areas = []
    frame_numbers = []

    # Process each frame
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            landmarks = predictor(gray, face)
            mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)], dtype=np.int32)
            area, hull = calculate_mouth_area(mouth_points)

            # Convert to real-world units
            real_mouth_area_cm2 = area / (pixels_per_cm ** 2)

            # Draw contours and text
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
            cv2.putText(frame, f'Pixel Area: {area:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Real Area: {real_mouth_area_cm2:.2f} cm²', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save data
            mouth_areas.append(area)
            real_mouth_areas.append(real_mouth_area_cm2)
            frame_numbers.append(idx)


    # Compile results into a DataFrame
    data = {
        "Frame Number": frame_numbers,
        "Mouth Area (px²)": mouth_areas,
        "Real Area (cm²)": real_mouth_areas
    }
    return pd.DataFrame(data)

# Main function
def main():
    # Bring the tkinter dialogs to the front
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # Ask user for frame folder
    frame_folder = filedialog.askdirectory(title="Select Folder with Frame Images")
    if not frame_folder:
        print("No folder selected. Exiting...")
        return


    # Process frames
    result_df = process_frames(frame_folder)

    if result_df is not None:
        # Save data to Excel
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        save_excel_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")], title="Save Data File")
        if save_excel_path:
            result_df.to_excel(save_excel_path, index=False)
            print(f"Data saved to {save_excel_path}")

if __name__ == "__main__":
    main()
