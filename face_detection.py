import cv2


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

while True:
    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.08,minNeighbors=5)

    for x,y,w,h in faces:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Galaxy",img)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break


video.relase()
cv2.destroyAllWindows()
