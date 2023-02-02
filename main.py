# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from djitellopy import Tello
import tensorflow as tf
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
print(tf.version)
me = Tello()


me.connect()
print(me.get_battery())
me.takeoff()
me.streamon()

while True:

    img = me.get_frame_read().frame

#face detection adapted from https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
    img = cv2.resize(img, (360, 240))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    faces = body_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("Image", img)

    cv2.waitKey(1)





