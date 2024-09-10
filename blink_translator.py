import cv2
import mediapipe as mp
import time
import math

# Initialize Mediapipe Face Mesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)

# Morse code dictionary
morse_dict = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z'
}

# Track blinks and translate to Morse code
morse_code = ""
blink_start_time = 0
blink_duration = 0
blink_threshold = 0.2  # Threshold for detecting blinks
dot_threshold = 0.2  # Duration for a dot (short blink)
dash_threshold = 0.6  # Duration for a dash (long blink)

def calculate_ear(landmarks, left_indices, right_indices):
    # EAR = (vertical distance between eyelids) / (horizontal distance)
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    left_vertical = distance(landmarks[left_indices[1]], landmarks[left_indices[5]])
    right_vertical = distance(landmarks[right_indices[1]], landmarks[right_indices[5]])
    
    left_horizontal = distance(landmarks[left_indices[0]], landmarks[left_indices[3]])
    right_horizontal = distance(landmarks[right_indices[0]], landmarks[right_indices[3]])

    ear_left = left_vertical / left_horizontal
    ear_right = right_vertical / right_horizontal

    return (ear_left + ear_right) / 2

def detect_blinks(landmarks):
    global blink_start_time, morse_code

    # Eye landmarks for left and right eye
    left_eye_indices = [33, 160, 159, 133, 153, 144]
    right_eye_indices = [362, 385, 386, 263, 373, 380]

    # Calculate EAR (Eye Aspect Ratio)
    ear = calculate_ear(landmarks.landmark, left_eye_indices, right_eye_indices)

    # Debugging: Print the EAR value to help adjust blink_threshold
    print(f"EAR: {ear}")

    if ear < blink_threshold:
        if blink_start_time == 0:
            blink_start_time = time.time()  # Start tracking blink duration
    else:
        if blink_start_time != 0:
            blink_duration = time.time() - blink_start_time

            # Detect whether the blink is a dot or a dash
            if blink_duration < dot_threshold:
                morse_code += "."
                print(f"Detected a dot: {morse_code}")
            elif blink_duration < dash_threshold:
                morse_code += "-"
                print(f"Detected a dash: {morse_code}")

            # Reset the blink start time
            blink_start_time = 0

    return morse_code


def translate_morse_code(morse_code):
    return morse_dict.get(morse_code, '')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the frame to RGB (Mediapipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe FaceMesh
    result = face_mesh.process(rgb_frame)

    # If face landmarks are detected, draw them and detect blinks
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            
            # Detect blinks and translate them to Morse code
            morse_code = detect_blinks(face_landmarks)

            # If a complete Morse code sequence is detected, translate it
            if len(morse_code) > 0 and time.time() - blink_start_time > 2:  # 2-second pause indicates end of a letter
                letter = translate_morse_code(morse_code)
                print(f"Morse code: {morse_code} -> {letter}")
                morse_code = ""  # Reset Morse code after translation

    # Display the frame with landmarks
    cv2.imshow('Morse Code Blinks', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
