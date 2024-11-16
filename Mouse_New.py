import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and Mediapipe Face Mesh
cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables to track blink state and cursor movement
blink_start_time = None
blink_duration = 0.3  # Blink threshold in seconds
last_cursor_x, last_cursor_y = 0, 0  # Initialize cursor position
cursor_smooth_factor = 0.1  # Smoothing factor for cursor movement

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Draw circles for face mesh landmarks
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Smooth the cursor movement (linear interpolation)
                screen_x = last_cursor_x + cursor_smooth_factor * (screen_x - last_cursor_x)
                screen_y = last_cursor_y + cursor_smooth_factor * (screen_y - last_cursor_y)

                pyautogui.moveTo(screen_x, screen_y)
                last_cursor_x, last_cursor_y = screen_x, screen_y

        # Eye landmarks for blinking detection (left and right eyes)
        left = [landmarks[145], landmarks[159]]  # Left eye landmarks
        right = [landmarks[374], landmarks[386]]  # Right eye landmarks

        # Draw circles for eye landmarks
        for landmark in left + right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Check for blink in left eye
        left_blink_detected = (left[0].y - left[1].y) < 0.005
        # Check for blink in right eye
        right_blink_detected = (right[0].y - right[1].y) < 0.005

        if left_blink_detected or right_blink_detected:
            if blink_start_time is None:
                blink_start_time = time.time()  # Start timer for blink

            # If blink duration exceeds threshold, perform right-click
            if time.time() - blink_start_time > blink_duration:
                pyautogui.rightClick()
                print("Right Click")
                blink_start_time = None  # Reset blink timer
                pyautogui.sleep(0.1)
        else:
            # If eyes are open, reset the blink timer
            blink_start_time = None

        # If the left blink condition is met, perform a left-click (original code)
        if left_blink_detected:
            pyautogui.click()
            print("Left Click")
            pyautogui.sleep(0.1)

    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
