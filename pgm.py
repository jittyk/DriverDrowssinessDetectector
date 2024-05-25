import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
import time
import pygame
import pyttsx3

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Jincyjitty\Desktop\s7project\self\shape_predictor_68_face_landmarks.dat")

# Initialize pygame
pygame.mixer.init()

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Define a function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define a function to compute the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Define a function to compute the head pose
def head_pose(shape):
    image_pts = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    model_pts = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = (frame_width + frame_height) / 2
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    euler_angles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

    return euler_angles  # Return pitch, yaw, and roll angles

# Define constants
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for drowsiness detection
MAR_THRESHOLD = 0.55   # Mouth aspect ratio threshold for yawning detection
HEAD_DOWN_THRESHOLD = -15  # Head down angle threshold (degrees)
ALARM_DURATION_EYES_CLOSED = 4  # Duration in seconds for eyes closed violation
ALARM_DURATION_HEAD_DOWN = 15  # Duration in seconds for head down violation
FACE_WARNING_DURATION = 5  # Duration in seconds for no face detected warning
CONSEC_FRAMES = 20  # Number of consecutive frames for which the EAR must be below the threshold to trigger an alert

# Define the path to the alarm sound file
alarm_sound_path = r"C:\Users\Jincyjitty\Desktop\s7project\self\security-alarm-80493.mp3"

# Initialize variables
start_time_eyes_closed = None
start_time_head_down = None
start_time_no_face = None  # Variable to track the start time of no face detected warning
drowsy_frames = 0
yawn_frames = 0
prev_pitch = 0  # Initialize prev_pitch variable
prev_roll = 0   # Initialize prev_roll variable
nod_count = 0
shake_count = 0

# Start video capture
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_warning_given = False  # Flag to track if face warning is given

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    if len(rects) == 0 and not face_warning_given:
        # If no faces detected and warning not given yet, sound the alarm and give a warning
        cv2.putText(frame, "WARNING: No Face Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        pygame.mixer.music.load(alarm_sound_path)
        pygame.mixer.music.play()
        engine.say("Warning! No face detected.")
        engine.runAndWait()
        face_warning_given = True
        start_time_no_face = time.time()  # Record the start time of no face detected warning

    # Check if the duration of no face detected warning has exceeded FACE_WARNING_DURATION
    if start_time_no_face is not None and time.time() - start_time_no_face >= FACE_WARNING_DURATION:
        # Reset the warning flag after the specified duration
        face_warning_given = False

    # If faces are detected, reset the warning flag and start time
    if len(rects) > 0:
        face_warning_given = False
        start_time_no_face = None

    alarm_triggered = False  # Reset alarm flag for each frame

    for rect in rects:
        # Determine facial landmarks
        shape = predictor(gray, rect)
        shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        # Extract left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Extract mouth coordinates
        mouth = shape[48:68]

        # Calculate eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Compute the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # Calculate mouth aspect ratio
        mar = mouth_aspect_ratio(mouth)

        # Calculate head pose angles (pitch, yaw, and roll)
        pitch, yaw, roll = head_pose(shape)

        # Check for eyes closed violation
        if ear < EAR_THRESHOLD:
            if start_time_eyes_closed is None:
                start_time_eyes_closed = time.time()
            elif time.time() - start_time_eyes_closed >= ALARM_DURATION_EYES_CLOSED and not alarm_triggered:
                cv2.putText(frame, "WARNING: Eyes Closed Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.load(alarm_sound_path)
                pygame.mixer.music.play()
                alarm_triggered = True
                engine.say("Warning! Stay awake while driving.")
                engine.runAndWait()
        else:
            start_time_eyes_closed = None

        # Check for yawning violation
        if mar > MAR_THRESHOLD:
            yawn_frames += 1
            if yawn_frames >= CONSEC_FRAMES and not alarm_triggered:
                cv2.putText(frame, "WARNING: Yawning Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.load(alarm_sound_path)
                pygame.mixer.music.play()
                alarm_triggered = True
                engine.say("Warning! Don't yawn while driving.")
                engine.runAndWait()
        else:
            yawn_frames = 0

        # Check for head down violation
        if pitch < HEAD_DOWN_THRESHOLD:
            if start_time_head_down is None:
                start_time_head_down = time.time()
            elif time.time() - start_time_head_down >= ALARM_DURATION_HEAD_DOWN and not alarm_triggered:
                cv2.putText(frame, "WARNING: Head Down Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.load(alarm_sound_path)
                pygame.mixer.music.play()
                alarm_triggered = True
                engine.say("Warning! Head Down Detected!")
                engine.runAndWait()
        else:
            start_time_head_down = None

        # Track head nods and shakes
        if pitch < -15:  # Threshold for downward head nod
            if prev_pitch > 0:  # Previous pitch was positive (upward movement)
                nod_count += 1
        elif pitch > 15:  # Threshold for upward head nod
            if prev_pitch < 0:  # Previous pitch was negative (downward movement)
                nod_count += 1
        elif abs(roll) > 15:  # Threshold for head shake
            shake_count += 1

        prev_pitch = pitch
        prev_roll = roll

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check if any key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
