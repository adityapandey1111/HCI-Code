# by Aditya Kumar Pandey and group

import cv2
import time
import mediapipe as mp
import autopy
import numpy as np
import math

# initialize the variable of buffer window
width_cam = 640
height_cam = 480
frame_reduction = 100
smoothen = 6

p_time = 0
p_loca_x, p_loca_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, height_cam)
cap.set(4, width_cam)

# hand landmarks
mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# display size
width_screen, height_screen = autopy.screen.size()

# cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE) # Create window with autosize flag
# cv2.startWindowThread() # Start a thread to manage the window
# cv2.moveWindow("Image", -10000, -10000) # Move window out of the screen


while True:
    # Draw Hand Landmarks
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    if result.multi_hand_landmarks:
        lm_list = []
        x_list = []
        y_list = []
        for hand_lm in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_lm.landmark):
                height, width, channel = img.shape
                x, y = int(lm.x * width), int(lm.y * height)
                x_list.append(x)
                y_list.append(y)
                lm_list.append([id, x, y])
                mp_draw.draw_landmarks(img, hand_lm, mp_hand.HAND_CONNECTIONS)

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
            cv2.rectangle(
                img, (x_min - 20, y_min - 20),
                (x_max + 20, y_max + 20),
                (0, 255, 0), 2
            )
        # print(lm_list)
        if len(lm_list) > 15:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            # Frame Reduction
            cv2.rectangle(img, (frame_reduction, frame_reduction),
                          (width_cam - frame_reduction, height_cam - frame_reduction),
                          (255, 255, 0), 3)

            # Moving Mode - Index finger Up
            if y1 < lm_list[6][2] and y2 > lm_list[10][2]:
                # Convert Coordinates into mouse Coordinates
                x3 = np.interp(x1, (frame_reduction, width_cam - frame_reduction), (0, width_screen))
                y3 = np.interp(y1, (frame_reduction, height_cam - frame_reduction), (0, height_screen))

                # Smoothening
                c_loca_x = p_loca_x + (x3 - p_loca_x) / smoothen
                c_loca_y = p_loca_y + (y3 - p_loca_y) / smoothen

                # Move Mouse
                autopy.mouse.move(width_screen - c_loca_x, c_loca_y)
                cv2.circle(
                    img, (x1, y1), 10,
                    (255, 0, 255), cv2.FILLED
                )
                p_loca_x, p_loca_y = c_loca_x, c_loca_y

            # Selection Mode - Index and Middle fingers are Up
            elif y1 < lm_list[6][2] and y2 < lm_list[10][2]:

                # Find Length between Index and middle finger
                length = math.hypot(x2 - x1, y2 - y1)
                # If length < 40 - Mouse Click
                if length < 40:
                    cv2.circle(
                        img, (x1, y1), 10,
                        (0, 255, 0), cv2.FILLED
                    )
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)

    # Frame Per Sec
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(
        img, f'FPS: {int(fps)}', (10, 50),
        cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 0, 0), 2
    )
    cv2.imshow("Image", img)
    cv2.waitKey(1)