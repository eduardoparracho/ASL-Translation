import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def extract_hand_keypoints(video_path, output_dir, save_every_n_frames=1):
    """
    Extracts hand keypoints from a given .mkv video and saves them as .npy files.

    Parameters:
    - video_path (str): Path to the input .mkv video.
    - output_dir (str): Directory to save the keypoints.
    - save_every_n_frames (int): Save keypoints every N frames.

    Returns:
    - Saves a .npy file with extracted keypoints for each video.
    """

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_list = []

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        frame_count += 1
        if frame_count % save_every_n_frames != 0:
            continue  # Skip frames to reduce computation

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(frame_rgb)

        # Extract hand keypoints
        keypoints = np.zeros((2, 21, 3))  # Default to zeros (2 hands, 21 landmarks, 3D coordinates)
        
        if results.multi_hand_landmarks:  # Check if hands are detected
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < 2:  # Limit to 2 hands
                    keypoints[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        keypoints_list.append(keypoints.flatten())

    cap.release()
    hands.close()

    # Convert list to NumPy array
    keypoints_array = np.array(keypoints_list)

    # Save to .npy file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(video_path).replace('.mkv', '.npy'))
    np.save(output_file, keypoints_array)

    print(f"Saved keypoints to {output_file}")



# MediaPipe hand landmark connections for 2D plotting
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)  # Palm base connection
]


def visualize_hand_keypoints_2d(npy_file):
    """
    TODO: Fix "visualize_hand_keypoints_2d", only works inside a jupyter notebook, doesn't work in a script
    Enable interactive Matplotlib mode in Jupyter with the command:
    %matplotlib notebook  
    """ 

    keypoints = np.load(npy_file)  # Shape: (num_frames, 126)
    
    if keypoints.shape[1] != 126:
        print("Error: Expected shape (num_frames, 126) but got", keypoints.shape)
        return

    num_frames = keypoints.shape[0]
    keypoints = keypoints.reshape(num_frames, 2, 21, 3)  # (frames, hands, landmarks, x/y/z)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)  # Normalized coordinates (0 to 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Invert Y-axis to match image coordinates

    # Initialize hand dots and connection lines
    left_hand_dots, = ax.plot([], [], 'ro', markersize=5)  # Red for left hand
    right_hand_dots, = ax.plot([], [], 'bo', markersize=5)  # Blue for right hand
    left_hand_lines = [ax.plot([], [], 'r-')[0] for _ in HAND_CONNECTIONS]
    right_hand_lines = [ax.plot([], [], 'b-')[0] for _ in HAND_CONNECTIONS]

    def update(frame):
        ax.set_title(f"Frame: {frame}/{num_frames}")

        left_hand, right_hand = keypoints[frame, :, :, :2]  # Extract only X, Y

        # Update scatter points
        left_hand_dots.set_data(left_hand[:, 0], left_hand[:, 1])
        right_hand_dots.set_data(right_hand[:, 0], right_hand[:, 1])

        # Update lines
        for i, (start, end) in enumerate(HAND_CONNECTIONS):
            left_hand_lines[i].set_data([left_hand[start, 0], left_hand[end, 0]],
                                        [left_hand[start, 1], left_hand[end, 1]])
            right_hand_lines[i].set_data([right_hand[start, 0], right_hand[end, 0]],
                                         [right_hand[start, 1], right_hand[end, 1]])

        return left_hand_dots, right_hand_dots, *left_hand_lines, *right_hand_lines

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    return HTML(ani.to_jshtml())  # Display animation inside Jupyter


def normalize_keypoints(npy_file, save_path):
    """
    Loads keypoints from a .npy file, normalizes them relative to the wrist, 
    and scales to [-1, 1]. Saves the normalized keypoints to a new .npy file.
    !!! This is legacy code, it does not help training, it was just for visualizations - also only works on ipynb files!!!
    
    Parameters:
    - npy_file (str): Path to the input .npy file (contains keypoints).
    - save_path (str): Path to save the normalized keypoints.
    """
    # Load the keypoints from the .npy file (Shape: [num_frames, 126])
    keypoints = np.load(npy_file)  
    
    # Ensure the shape is correct (num_frames, 126)
    if keypoints.shape[1] != 126:
        print("Error: Expected shape (num_frames, 126) but got", keypoints.shape)
        return

    num_frames = keypoints.shape[0]
    keypoints = keypoints.reshape(num_frames, 2, 21, 3)  # Reshape to (frames, hands, 21 points, x/y/z)

    # Normalize each frame
    for i in range(num_frames):
        for hand in range(2):  # Loop over left (0) and right (1) hands
            wrist = keypoints[i, hand, 0, :2]  # Get the wrist X, Y coordinates (landmark 0)
            
            # Shift all keypoints to make wrist the origin
            keypoints[i, hand, :, :2] -= wrist  # Subtract wrist position from all points
            
            # Scale all X, Y coordinates to [-1, 1]
            max_value = np.max(np.abs(keypoints[i, hand, :, :2]))  # Max value in X, Y
            keypoints[i, hand, :, :2] /= max_value  # Normalize to [-1, 1]

    # Flatten back to shape (num_frames, 126) and save
    keypoints = keypoints.reshape(num_frames, 126)
    np.save(save_path, keypoints)  # Save normalized keypoints
    
    print(f"Normalized keypoints saved to: {save_path}")
