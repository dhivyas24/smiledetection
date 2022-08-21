from random import randrange
import cv2

trained_datafront = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_datasmile = cv2.CascadeClassifier('smile.xml')
#img = cv2.imread('download.jfif')
web = cv2.VideoCapture(0)
while True:
    frameread , img = web.read()

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    facecordinates = trained_datafront.detectMultiScale(grayscale)
    

    for(x,y,w,h) in facecordinates:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        thefaces=img[y:y+h,x:x+h]
        facegrayscale = cv2.cvtColor(thefaces, cv2.COLOR_BGR2GRAY)
        smiles = trained_datasmile.detectMultiScale(facegrayscale, 1.7, 22)
        if(len(smiles)>0):
            cv2.putText(img,"Smiling",(x,y-50),cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),3,cv2.LINE_AA)
        else:
            cv2.putText(img,"NotSmiling",(x,y-50),cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),3,cv2.LINE_AA)    
          
    cv2.imshow('Hello',img)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
web.release()


