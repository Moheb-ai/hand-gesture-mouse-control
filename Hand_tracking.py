print('Hand Tracking program started.')
import cv2
import mediapipe as mp
import pyautogui
import math

def had_gesture_mouse_control():
    cam = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    screen_width, screen_height = pyautogui.size()

    clicking = False
    right_clicking = False
    pinch_threshold = 0.03

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * frame.shape[1])
                y = int(index_finger.y * frame.shape[0])

                screen_x = screen_width * index_finger.x
                screen_y = screen_height * index_finger.y
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                # ================= LEFT CLICK (Thumb + Index) =================
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                dx = thumb_tip.x - index_tip.x
                dy = thumb_tip.y - index_tip.y
                distance_left = math.sqrt(dx*dx + dy*dy)

                if distance_left < pinch_threshold and not clicking:
                    pyautogui.click()
                    clicking = True
                elif distance_left >= pinch_threshold and clicking:
                    clicking = False


                # ================= RIGHT CLICK (Thumb + Middle) =================
                middle_tip = hand_landmarks.landmark[12]

                dx2 = thumb_tip.x - middle_tip.x
                dy2 = thumb_tip.y - middle_tip.y
                distance_right = math.sqrt(dx2*dx2 + dy2*dy2)

                if distance_right < pinch_threshold and not right_clicking:
                    pyautogui.rightClick()
                    right_clicking = True
                elif distance_right >= pinch_threshold and right_clicking:
                    right_clicking = False

        cv2.imshow("Hand Gesture Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    had_gesture_mouse_control()