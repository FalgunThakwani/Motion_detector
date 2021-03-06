import cv2 ,time,pandas
from datetime import datetime

video=cv2.VideoCapture(0)
first_frame=None
status_list=[None,None]
time=[]
df=pandas.DataFrame(columns=["Start","End"])

while True:
    status=0
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    threshold_frame=cv2.threshold(delta_frame,40,255,cv2.THRESH_BINARY)[1]
    threshold_frame=cv2.dilate(threshold_frame,None,iterations=2)
    (cnts,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<5000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        status=1
    status_list.append(status)


    if status_list[-1]==1 and status_list[-2]==0:
        time.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        time.append(datetime.now())


    cv2.imshow("first_frame",first_frame)
    cv2.imshow("gray",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("threshold_frame",threshold_frame)
    cv2.imshow("colorframe",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            time.append(datetime.now())
        break

for i in range(0,len(time),2):
    df=df.append({"Start":time[i],"End":time[i+1]},ignore_index=True)

df.to_csv("Times.csv")
print(gray)
print(delta_frame)

print(time)

video.release()
cv2.destroyAllWindows()
