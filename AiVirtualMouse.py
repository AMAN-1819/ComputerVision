import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  # Lower resolution for faster processing
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]

            index_x = int(index_finger_tip.x * frame_width)
            index_y = int(index_finger_tip.y * frame_height)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), thickness=-1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), thickness=-1)

            # Calculate the screen position
            target_x = screen_width / frame_width * index_x
            target_y = screen_height / frame_height * index_y

            # Move mouse
            pyautogui.moveTo(target_x, target_y)

            # Click if close
            if abs(index_y - thumb_y) < 20:
                pyautogui.click()

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
