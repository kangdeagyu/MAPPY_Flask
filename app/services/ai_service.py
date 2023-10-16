import cv2
from keras.models import load_model
import numpy as np
from flask import Flask
import pandas as pd
from PIL import Image

class AI_Service:
    def __init__(self):
        self.model = load_model('app/static/best_face_cnn_model.h5')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def AI_predict(self, image_file, target_size=(128, 128), padding_size=(130, 130)):
        try:
            # 이미지 불러오기 (PIL Image -> Numpy Array)
            img_pil = Image.open(image_file)
            img = np.array(img_pil)

            # BGR GRAY 색상 변환 (PIL은 RGB 순서이므로 RGB->BGR 변환 후 GRAY 색상 변환)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 얼굴 찾기
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_crop = img[y:y+h,x:x+w]
                face_resized=cv2.resize(face_crop,target_size)

                padded_face=np.zeros((padding_size[1], padding_size[0],3),dtype=np.uint8)
                x_offset=(padding_size[0]-target_size[0])//2
                y_offset=(padding_size[1]-target_size[1])//2

                padded_face[y_offset:y_offset+target_size[1],x_offset:x_offset+target_size[0]]=face_resized

                padded_face_gray=cv2.cvtColor(padded_face,cv2.COLOR_BGR2GRAY)

            train=np.zeros(1*130*130*1,dtype=np.int32).reshape(1 ,130 ,130) 
            img=np.array(padded_face_gray,dtype=np.int32)
            train[0,:,:]=img
            train=train.reshape(-1,130*130)

            df=pd.DataFrame(train)

            width, height, channel = 130 , 130 , 1 

            x_train=df.values
            x_train=x_train.reshape((x_train.shape[0],width,height,channel))

            X=(x_train-127.5)/127.5

            pred=self.model.predict(X)

            classes=['10대','20대','30대','40대','50대','60대','70대']
            
            return classes[np.argmax(pred)]
        except Exception as e:
            print('error:',e)
            err=['인식오류']
            return err
