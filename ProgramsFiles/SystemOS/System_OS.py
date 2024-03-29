# by Aditya Kumar Pandey and group

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyautogui

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('ProgramsFiles\SystemOS\mp_hand_gesture')

# Load class names
f = open('ProgramsFiles\SystemOS\gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE) # Create window with autosize flag
cv2.startWindowThread() # Start a thread to manage the window
cv2.moveWindow("Image", -10000, -10000) # Move window out of the screen

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Lock screen if index finger up gesture is detected
            if className == 'rock':
                os.system("rundll32.exe user32.dll,LockWorkStation")
                break
            
            # Shut Down if thumbs up gesture is detected
            elif className == 'thumbs up':
                os.system("shutdown /s /t 1")
                break
            
            # Restart if five fingers gesture is detected together
            elif className == 'stop':
                os.system("shutdown /r /t 1")
                break

            # screenshot using fist
            elif className == 'fist':
                pyautogui.hotkey('win','PrtScr')
                break

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    # cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()