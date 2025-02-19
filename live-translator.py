import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained model
MODEL_PATH = "alphabet/alphabet_modelv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels (A-Z + special characters)
LABELS = {i: chr(65 + i) for i in range(26)}  # A-Z

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image horizontally for mirror effect - !This needs further testing!

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)

            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If hand detected, make a prediction
    if len(keypoints) == 42:  # (21 keypoints × (x, y))
        keypoints = np.array(keypoints).reshape(1, -1)  # Reshape for model
        prediction = model.predict(keypoints)
        letter = LABELS[np.argmax(prediction)]

        # Display result on top left
        cv2.putText(frame, f"Predicted: {letter}", (75, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("ASL Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()