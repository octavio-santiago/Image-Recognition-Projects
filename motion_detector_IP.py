import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import datetime
from PIL import Image
import requests
from io import BytesIO
import imutils

#python 2.7

#import cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
ubody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

#security cams urls
#https://www.insecam.org/
url = 'http://187.84.58.132/jpg/image.jpg'
url1 = 'http://189.85.187.122/cgi-bin/viewer/video.jpg'
url3 = 'http://222.100.79.51:50000/SnapshotJPEG?Resolution=640x480&amp;amp;Quality=Clarity&amp;amp;1515330100'
url4='http://121.7.35.213/SnapshotJPEG?Resolution=640x480&amp;amp;Quality=Clarity&amp;amp;1515330296'
url2 = 'http://187.109.215.84:8080/mjpg/video.mjpg?COUNTER'
url5 = 'http://138.185.163.76:84/webcapture.jpg?command=snap&amp;channel=1?1515342434'
url6 ='http://201.87.210.51:81/videostream.cgi?user=admin&pwd='
sala = 'http://150.161.159.115/webcapture.jpg?command=snap&amp;channel=1?0'
predioTrab = 'http://189.90.8.42:8084/cgi-bin/camera?resolution=640&amp;amp;quality=1&amp;amp;Language=0&amp;amp;1515343360'
casinha = 'http://177.91.127.94:8090/webcapture.jpg?command=snap&amp;channel=1?1515343432'
trab = 'http://131.72.69.106:8001/webcapture.jpg?command=snap&amp;channel=1?1515343475'
frentecasa='http://131.0.248.185:8000/webcapture.jpg?command=snap&amp;channel=1?1515343000'
trator = 'http://177.129.44.134:8080/webcapture.jpg?command=snap&amp;channel=1?1515343016'
url = 'http://58.94.98.44/nphMotionJpeg?Resolution=640x480&Quality=Standard'
url = 'http://145.53.212.190:8001/mjpg/video.mjpg'
url = 'http://98.151.134.48:82/nphMotionJpeg?Resolution=640x480&Quality=Clarity'
url = 'http://77.243.103.105:8081/mjpg/video.mjpg'

min_area = 500
count = 0

contador = 0
cam = cv2.VideoCapture(url)
firstFrame = None
#looping
while True:
    count = count + 1
    grabbed, frame = cam.read()
    text = "Parado"
    if not grabbed:
        print ('deu erro - frame nao pode ser capturado')
        break
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    faces = face_cascade.detectMultiScale(frame,1.8,5)
    body = body_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    ubody = ubody_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x1,y1,w1,h1) in body:
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,255),2)
        #print "Found {0} body!".format(len(body))
        #cv2.SaveImage('body.jpg',img)
    for (x2,y2,w2,h2) in ubody:
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),2)
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)    

    #detectar corpos que se mexem
    body2 = body_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5)

    for (x3,y3,w3,h3) in body2:
        cv2.rectangle(frame,(x3,y3),(x3+w3,y3+h3),(0,0,0),2)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Em movimento"
        contador = contador + 1
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow('test',frame)
    if ord('q') == cv2.waitKey(10):
        exit(0)
        break


#Aplicar na camera----
#cap = cv2.VideoCapture(1)
#while True:
    #ter,img=cap.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(img,1.3,5)
    #for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #cv2.imshow('img',img)
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
        #break

#print (contador)
cam.release()        
cv2.destroyAllWindows()
