import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from mediapipe.framework.formats import landmark_pb2


# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def normalize_landmarks(face_landmarks):
    nose_tip = face_landmarks.landmark[4] # nose
    chin = face_landmarks.landmark[152]
    
    # Calculate the scaling factor as the Euclidean distance between nose tip and chin
    scale = np.sqrt((nose_tip.x - chin.x) ** 2 + (nose_tip.y - chin.y) ** 2 + (nose_tip.z - chin.z) ** 2)
    
    normalized_landmarks = []
    for landmark in face_landmarks.landmark:
        normalized_landmarks.append(
            landmark_pb2.Landmark(
                x=(landmark.x - nose_tip.x) / scale,
                y=(landmark.y - nose_tip.y) / scale,
                # z=(landmark.z - nose_tip.z) / scale, # You can uncomment this if you need the z-coordinate
            )
        )
    return normalized_landmarks

# Define a function to determine face orientation
def get_face_orientation(normalized_landmarks):
    forehead = normalized_landmarks[10]
    nose_tip = normalized_landmarks[4]
    chin = normalized_landmarks[152]
    left_eye = normalized_landmarks[234]
    right_eye = normalized_landmarks[454]

    # Calculate the vertical distance between nose tip and chin
    vertical_upper_distance = nose_tip.y - chin.y
    vertical_lower_distance = forehead.y - nose_tip.y

    # Calculate the horizontal distance between left and right eye
    horizontal_distance_right = nose_tip.x - right_eye.x
    horizontal_distance_left = nose_tip.x - left_eye.x

    print(f'Vertical Upper Distance: {vertical_upper_distance}')
    print(f'Vertical Lower Distance: {vertical_lower_distance}')
    print(f'Horizontal Distance Right: {horizontal_distance_right}')
    print(f'Horizontal Distance Left: {horizontal_distance_left}')

    if vertical_upper_distance > -0.8 and vertical_upper_distance < -0.6 and vertical_lower_distance < -1.20 and vertical_lower_distance > -1.40:  # Adjust the vertical threshold as needed
        return 'Facing Down'
    elif vertical_upper_distance > -0.99 and vertical_upper_distance < -0.95 and vertical_lower_distance < -0.4 and vertical_lower_distance > -0.55:  # Adjust the vertical threshold as needed
        return 'Facing Up'
    elif horizontal_distance_right < -1 and horizontal_distance_right > -1.26 and horizontal_distance_left < 0.14 and horizontal_distance_left > -0.52:  # Adjust the horizontal threshold as needed
        return 'Facing Right'
    elif horizontal_distance_right > -0.07 and horizontal_distance_right < 0.6 and horizontal_distance_left > 0.9 and horizontal_distance_left < 1.36:  # Adjust the horizontal threshold as needed
        return 'Facing Left'
    else:
        return 'Facing Forward'

# Create directories for dataset if they do not exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save frames with a unique name in the specified class folder
def save_frame(frame, sDirektoriData, sKelas):
    # Ensure the directory exists
    class_dir = os.path.join(sDirektoriData, sKelas)
    ensure_directory_exists(class_dir)
    
    # Create a unique filename for each frame
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    filepath = os.path.join(class_dir, filename)
    
    # Save the frame
    cv2.imwrite(filepath, frame)
    print(f"Saved: {filepath}")


def create_dataset_with_landmarks(sDirektoriData, sKelas, NoKamera, FrameRate):
    # Initialize video capture
    cap = cv2.VideoCapture(NoKamera)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5) as face_mesh:

        prev_frame_time = 0  # Initialize the previous frame time variable

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

             # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            fps_text = f'FPS: {fps:.2f}'
            
            # Process the frame with MediaPipe FaceMesh
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Normalize the landmarks
                    normalized_landmarks = normalize_landmarks(face_landmarks)
                    
                    # Get face orientation
                    orientation = get_face_orientation(normalized_landmarks)

                    # Draw the face landmarks on the original image
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                    # Create a protobuf NormalizedLandmarkList
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    face_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                        for landmark in normalized_landmarks
                    ])

                    # Optionally, print out the landmarks
                    print(face_landmarks_proto)

                    # Save the frame with drawn landmarks
                    save_frame(frame, sDirektoriData, f"{sKelas}_original")

            # Put FPS text on the frame
            cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('MediaPipe FaceMesh', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


# Call the function to create your dataset
sDirektoriData = "d:\\1 PraTA\\Dataset\\KursiRoda1"
sKelas = "Mundur"
NoKamera = 0
FrameRate = 5
create_dataset_with_landmarks(sDirektoriData, sKelas, NoKamera, FrameRate)
