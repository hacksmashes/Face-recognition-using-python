import cv2
import numpy
import os
import imutils

haar_file="D:\\Python\\AI\\Algorithm\\haarcascade_frontalface_default.xml"   # give your directory of XML file      
face_cascade=cv2.CascadeClassifier(haar_file)
datasets="D:\\Python\\AI\\datasets"                                     # give the correct directory for dataset folder

(images,labels,names,id)=([],[],{},0)
cam=cv2.VideoCapture(0)                              

for (subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath + '/' + filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1

(width,height)=(130,100)

(images,labels)=[numpy.array(lis) for lis in [images,labels]]
#print(images,labels)

model=cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
print("Training completed")


cnt=0
while True:
    img=cam.read()[1]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<800:
            cv2.putText(img,"%s-%.0f"% (names[prediction[0]],prediction[1]),(x+0,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,255),2)
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(img,"unknown",(x-10,y-10),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,255),2) 
            if (cnt>100):
                print("Unknown Person")   
                cv2.imwrite("input.png",img)
                cnt=0
    cv2.imshow("Face Recognition",img)
    key=cv2.waitKey(10)
    if key==27:
        break        
    

cam.release()
cv2.destroyAllWindows()    
