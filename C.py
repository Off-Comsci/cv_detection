import cv2
import idtoname
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors,minSize=(55, 55))
        coords=[]
        for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                id,con= clf.predict(gray[y:y+h,x:x+w])

                names=idtoname.ID_To_Name(id)
                # names=id
                cons = " {0}%".format(round(100 - con))
                print(str(id)+':'+str(con))
                    
                if con <= 80 :
                    cv2.putText(img,str(names)+str(cons),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                else :
                    cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                    
                coords=[x,y,w,h]
        return img,coords 
        
def detect(img,faceCascade,img_id,clf):
        img,coords=draw_boundary(img,faceCascade,1.1,10,(0,0,255),clf)
        if len(coords)==4 :
                result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]       
        return img

img_id=0
cap = cv2.VideoCapture(0)

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while (True):
        ret,frame = cap.read()
        frame=detect(frame,faceCascade,img_id,clf)
        cv2.imshow('frame',frame)
        img_id+=1
        if(cv2.waitKey(1) & 0xFF== ord('x')):
            break
cap.release()
cv2.destroyAllWindows()


