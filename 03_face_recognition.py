import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def blur_img(img, factor = 20):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)

    #ensure the shape of the kernel is odd
    if kW % 2 == 0: kW = kW - 1
    if kH % 2 == 0: kH = kH - 1

    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> loze: id=1,  etc
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
names = ['None', 'winter', 'karina', 'ningning', 'giselle']
filename=input()
# fourcc=cv2.VideoWriter_fourcc(*'DIVX')
# out=cv2.VideoWriter('output.avi',fourcc,25,(640,480))
# Initialize and start realtime video capture
cam = cv2.VideoCapture('{0}.mp4'.format(filename)) #input형태로 받기
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.01*cam.get(3)
minH = 0.01*cam.get(4)

while True:
    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            blurred_face = blur_img(detected_face, factor = 3)
            img[y:y+h, x:x+w] = blurred_face

    # out.write(img)
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
# out.release()
cam.release()
cv2.destroyAllWindows()
