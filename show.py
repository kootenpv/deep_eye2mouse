import time
import just
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

from pynput.mouse import Button, Controller

mouse = Controller()

it = 0
until_val = 0
t1 = time.time()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Display the resulting frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        until_val = it + 10000
    t2 = time.time()
    if t2 > t1 + 0.1 and it < until_val:
        cv2.imwrite("/Users/pascal/tracktrack/output.png", frame)
        prediction = just.read("/Users/pascal/tracktrack/output.json", no_exist=None)
        if prediction is not None:
            x, y = prediction
        mouse.position = (x, y)
        it += 1
        t1 = t2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
