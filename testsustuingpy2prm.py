import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Define a function to normalize landmarks 
def normalize_landmarks(face_landmarks):
    nose_tip = face_landmarks.landmark[4] #nose
    chin = face_landmarks.landmark[152]
    
    # Calculate the scaling factor as the Euclidean distance between nose tip and chin
    scale = np.sqrt((nose_tip.x - chin.x)**2 + (nose_tip.y - chin.y)**2 + (nose_tip.z - chin.z)**2)
    
    normalized_landmarks = []
    for landmark in face_landmarks.landmark:
        normalized_landmarks.append(
            landmark_pb2.Landmark(
                x=(landmark.x - nose_tip.x) / scale,
                y=(landmark.y - nose_tip.y) / scale,
                #z=(landmark.z - nose_tip.z) / scale
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

# Use MediaPipe FaceMesh
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        
        # Create a black image with the same dimensions as your video frame.
        black_image = np.zeros_like(frame)

        # Convert the BGR image to RGB and process it with MediaPipe FaceMesh.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw face landmarks of each face.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                normalized_landmarks = normalize_landmarks(face_landmarks)

                # Modify this line to draw on black_image instead of frame
                mp_drawing.draw_landmarks(black_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Determine face orientation
                orientation = get_face_orientation(normalized_landmarks)
                cv2.putText(black_image, f'Orientation: {orientation}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('MediaPipe FaceMesh', black_image) 

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
