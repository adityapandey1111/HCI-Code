# by Aditya Kumar Pandey and group

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)  # Checks for camera

mpHands = mp.solutions.hands  # detects hand/finger
hands = mpHands.Hands()  # complete the initialization configuration of hands
mpDraw = mp.solutions.drawing_utils

brightness = 0
brightbar = 400
brightper = 0

cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE) # Create window with autosize flag
cv2.startWindowThread() # Start a thread to manage the window
cv2.moveWindow("Image", -10000, -10000) # Move window out of the screen

while True:
    success, img = cap.read()  # If camera works capture an image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to rgb

    # Collection of gesture information
    results = hands.process(imgRGB)  # completes the image processing.

    lmList = []  # empty list
    if results.multi_hand_landmarks:  # list of all hands detected.
        # By accessing the list, we can get the information of each hand's corresponding flag bit
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):  # adding counter and returning it
                # Get finger joint points
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        # getting the value at a point
                    # x           #y
        x1, y1 = lmList[4][1], lmList[4][2]  # thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # index finger
        # creating circle at the tips of thumb and index finger
        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # create a line b/w tips of index finger and thumb

        length = hypot(x2 - x1 - 30, y2 - y1 - 30)  # distance b/w tips using hypotenuse
        # from numpy we find our length, by converting hand range in terms of brightness range i.e. between 0-255
        brightness = int(np.interp(length, [30, 350], [0, 255]))
        brightbar = int(np.interp(length, [30, 350], [400, 150]))
        brightper = int(np.interp(length, [12, 350], [0, 100]))
        sbc.set_brightness(int(brightper))
        
        print(brightness, int(length))

        # Hand range 30 - 350
        # Brightness range 0-255
        # creating brightness bar for brightness level
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255),4)  # vid ,initial position ,ending position ,rgb ,thickness
        cv2.rectangle(img, (50, int(brightbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(brightper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
        # tell the brightness percentage ,location,font of text,length,rgb color,thickness
    cv2.imshow('Image', img)  # Show the video
    if cv2.waitKey(1) & 0xff == ord(' '):  # By using spacebar delay will stop
        break

cap.release()  # stop cam
cv2.destroyAllWindows()  # close window 