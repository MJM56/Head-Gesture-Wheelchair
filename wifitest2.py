# Import necessary libraries
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import socket
import datetime
import time 

# Additional imports
import copy

host = "192.168.4.1" # Set to ESP32 Access Point IP Address
port = 80

kelasTemp = 'Start'
print("Global kelasTemp initialized:", kelasTemp)

def adjust_brightness(image, target_brightness=125):
    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the average brightness of the image
    avg_brightness = np.mean(hsv[:, :, 2])
    # Calculate the scaling factor to adjust the brightness
    scale_factor = target_brightness / avg_brightness
    # Apply the scaling factor to the Value channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale_factor, 0, 255).astype(np.uint8)
    # Convert back to BGR color space
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image

def preprocess_landmarks(landmarks):
    # Flatten the landmark points into a list or an array
    # Make sure to only take the first 468 landmarks if that's the expected number
    flattened_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[:468]]).flatten()
    expected_size = 468 * 3  # Adjust this value if the model expects a different number of landmarks
    if len(flattened_landmarks) != expected_size:
        raise ValueError(f"Expected {expected_size} values, but got {len(flattened_landmarks)}")
    # Reshape the landmarks as needed for your model input
    processed_landmarks = flattened_landmarks.reshape(-1, expected_size)
    return processed_landmarks

def cropped_image_normal(frame, landmarks):
    # Calculate the bounding box around the landmarks
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    H, W, _ = frame.shape
    x_min = int(x_min * W)
    x_max = int(x_max * W)
    y_min = int(y_min * H)
    y_max = int(y_max * H)

    # Find the longest side to make the bounding box a square
    width = x_max - x_min
    height = y_max - y_min
    longest_side = max(width, height)

    # Calculate new x_min, x_max, y_min, y_max to make the crop a square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - longest_side // 2)
    x_max = min(W, x_center + longest_side // 2)
    y_min = max(0, y_center - longest_side // 2)
    y_max = min(H, y_center + longest_side // 2)

    # Crop the image
    cropped_image = frame[y_min:y_max, x_min:x_max]
    return cropped_image


def cropped_image_black(frame, landmarks):
    # Calculate the bounding box around the landmarks
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    H, W, _ = frame.shape
    x_min = int(x_min * W)
    x_max = int(x_max * W)
    y_min = int(y_min * H)
    y_max = int(y_max * H)

    # Find the longest side to make the bounding box a square
    width = x_max - x_min
    height = y_max - y_min
    longest_side = max(width, height)

    # Calculate new x_min, x_max, y_min, y_max to make the crop a square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - longest_side // 2)
    x_max = min(W, x_center + longest_side // 2)
    y_min = max(0, y_center - longest_side // 2)
    y_max = min(H, y_center + longest_side // 2)

    # Crop the image
    cropped_black = frame[y_min:y_max, x_min:x_max]

    return cropped_black


def Klasifikasi(Image, ModelCNN):
    X = []
    img = copy.deepcopy(Image)
    img = cv2.resize(img, (250, 250))
    img = np.asarray(img) / 255
    X.append(img)
    X = np.array(X)
    X = X.astype('float32')
    start_time = time.time()  # Start time for inference
    hs = ModelCNN.predict(X, verbose=0)
    #print("Inference time:", end_time - start_time, "seconds")
    idx = -1
    if hs.max() > 0.5:
        idx = np.argmax(hs)
    end_time = time.time()  # End time for inference
        #print("Raw predictions:", hs)
    return idx


def PredictFaceMesh(NoKamera, LabelKelas):
    ModelCNN = load_model('D:/1 PraTA/Code/CNN/model.h5')
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(NoKamera)

    PrevIdx = -1  # Define PrevIdx before use
    counter = 0  # Define counter before use
    prev_frame_time = time.time()  # Initialize prev_frame_time before the loop
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            new_frame_time = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = adjust_brightness(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create a black image with the same dimensions as the frame
            black_image = np.zeros_like(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face landmarks on the original image
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
                    )

                    # Draw the face landmarks on the black image
                    mp_drawing.draw_landmarks(
                        black_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
                    )
                            # Draw specific landmarks with custom colors
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])

                    # Assuming classify_landmarks is the correct function to call
                    # and that it internally calls preprocess_landmarks
                    predicted_label = preprocess_landmarks(face_landmarks)
                    cropped_black = cropped_image_black(image, face_landmarks)

                    idx = Klasifikasi(cropped_black, ModelCNN)
                    x = 50
                    y = 50
                    # Check if the predicted class has changed
                    if idx >= 0 and idx != PrevIdx:
                    # Get current time with microseconds
                        current_time_with_microseconds = datetime.datetime.now()
                    # Format the time string to include milliseconds (first 3 digits of the microseconds)
                        current_time = current_time_with_microseconds.strftime("%Y-%m-%d %H:%M:%S") + "." + str(current_time_with_microseconds.microsecond // 1000)
                        print(f"Timestamp: {current_time}. Class changed to: {LabelKelas[idx]}")
                        print("Index:", idx)
                        PrevIdx = idx

                    if idx == 0 and idx != PrevIdx:
                        counter += 1
                        PrevIdx = idx

                    #print("Index:", idx)
                    if idx >= 0:
                        #print("Drawing label:", LabelKelas[idx])
                        cv2.putText(image, LabelKelas[idx], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 3)

                        # Create a socket connection
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            # Connect to the ESP32 server
                            s.connect((host, port))

                            if LabelKelas[idx]=='Kanan':
                                arah = 'E\n'
                                kecepatan = 250
                                message = f"{arah},{kecepatan}"
                                s.send(arah.encode('utf-8'))
                            elif LabelKelas[idx]=='Kiri':
                                arah = 'A\n'
                                kecepatan = 250
                                message = f"{arah},{kecepatan}"
                                s.send(arah.encode('utf-8'))
                            elif LabelKelas[idx]=='Maju':
                                arah = 'B\n'
                                kecepatan = 250
                                message = f"{arah},{kecepatan}"
                                s.send(arah.encode('utf-8'))
                            elif LabelKelas[idx]=='Mundur':
                                arah = 'D\n'
                                kecepatan = 250
                                message = f"{arah},{kecepatan}"
                                s.send(arah.encode('utf-8'))
                            elif LabelKelas[idx]=='Stop':
                                arah = 'C\n'
                                kecepatan = 0
                                message = f"{arah},{kecepatan}"
                                s.send(arah.encode('utf-8'))
            
             # Calculate FPS
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Display FPS on frame
            fps_text = "FPS: {:.2f}".format(fps)
            #Calculate text width & height to position text at top right corner
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = image.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
            text_y = text_size[1] + 10  # 10 pixels from the top edge

            cv2.putText(image, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 0), 3)
            cv2.putText(black_image, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 0), 3)
            # Display the black image with landmarks
            cv2.imshow('Black Image with Landmarks', black_image)

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# Define class labels for classification
LabelKelas = ["Kanan", "Kiri", "Maju", "Mundur", "Stop"]

# Call the function with webcam index and class labels
PredictFaceMesh(0, LabelKelas)