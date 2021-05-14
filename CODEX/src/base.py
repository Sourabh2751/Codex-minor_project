import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={"name":1}
with open("label.pickle",'rb') as f:
  op_labels=pickle.load(f)
  labels = {v:k for k,v in op_labels.items()}

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
      
      roi_gray=gray[y:y+h,x:x+w]
      roi_color=frame[y:y+h,x:x+w]

      #recognizer
      id_, conf=recognizer.predict(roi_gray)
      if conf>=4 and conf<=85:
       #showing name
       font=cv2.FONT_HERSHEY_SIMPLEX
       name=labels[id_]
       color=(255,255,255)
       stroke=2
       cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)



      img_item="my_image.png"
      cv2.imwrite(img_item,roi_color)
      #adding rectangle
      color=(255,0,0)
      stroke=2
      end_x=x+w
      end_y=y+h
      cv2.rectangle(frame,(x,y),(end_x,end_y),color,stroke)
      

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
 