import cv2
import mediapipe as mp

# Initialize mediapipe hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read video")
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                z = landmark.z

                # Do something with the landmark coordinates
                # (e.g., classify hand gesture based on landmark positions)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Define a function to classify hand gestures based on x and y coordinates
threshold1 = 100  # Adjust this threshold for distinguishing x-coordinate for numbers 0 to 9
threshold2 = 200  # Adjust this threshold for distinguishing y-coordinate for numbers 0 to 9

threshold3 = 300  # Adjust this threshold for distinguishing x-coordinate for letters A, B, and C
threshold4 = 400  # Adjust this threshold for distinguishing y-coordinate for letters A, B, and C

def classify_hand_gesture(x, y):
    # Perform classification logic here
    # You can use if-else statements or a machine learning model to classify the gestures
    # Example code for classifying numbers 0 to 9 and all alphabets

    # Classify numbers 0 to 9
    if x < threshold1 and y < threshold2:
        return "0"
    elif x < threshold1 and y > threshold2:
        return "1"
    elif x > threshold1 and y < threshold2:
        return "2"
    # Add more conditions for numbers 3 to 9

    # Classify alphabets
    if x < threshold3 and y < threshold4:
        return "A"
    elif x < threshold3 and y > threshold4:
        return "B"
    elif x > threshold3 and y < threshold4:
        return "C"
    # Add more conditions for other alphabets

# Within the existing code...
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read video")
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with mediapipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                z = landmark.z

                # Classify hand gesture based on landmark coordinates
                gesture = classify_hand_gesture(x, y)

                # Do something with the classified gesture (e.g., display it on the image)
                cv2.putText(image, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
