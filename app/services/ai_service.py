import cv2
from keras.models import load_model
import numpy as np
from flask import Flask
import dlib
import pandas as pd
from PIL import Image

class AI_Service:
    def __init__(self):
        self.model = load_model('app/static/64-256_256-2.h5')
        # Dlib 모델 불러오기
        self.face_cascade = dlib.get_frontal_face_detector()

    def AI_predict(self, image_file, target_size=(128, 128), padding_size=(182, 182)):
        try:
            # 이미지 불러오기 (Image -> Numpy Array)
            img_pil = Image.open(image_file)
            img = np.array(img_pil)

            # BGR GRAY 색상 변환 (PIL은 RGB 순서이므로 RGB->BGR 변환 후 GRAY 색상 변환)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 얼굴 찾기 
            rects = self.face_cascade(gray)

            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.width()
                h = rect.height()

                # 얼굴 영역 자르기 
                face_crop = img[y:y+h,x:x+w]

                # 잘라낸 면의 크기를 목표 크기로 조정 
                face_resized=cv2.resize(face_crop,target_size)

            # 학습된 데이터와 맞게 밝기 설정
            adjusted_image = face_resized.astype(float) - np.mean(face_resized) + 133.41081377158957

            adjusted_image[adjusted_image < 0] = 0   # 음수 값은 0으로 설정
            adjusted_image[adjusted_image > 255] = 255   # 초과하는 값은 최댓값인 255로 설정

            # 학습된 데이터와 맞게 패딩 설정
            padded_face=np.zeros((padding_size[1],padding_size[0],3),dtype=np.uint8)
    
            x_offset=(padding_size[0]-target_size[0])//2
            y_offset=(padding_size[1]-target_size[1])//2

            padded_face[y_offset:y_offset+target_size[1],x_offset:x_offset+target_size[0]]=adjusted_image

            # 그레이스케일 변경
            padded_face_gray=cv2.cvtColor(padded_face,cv2.COLOR_BGR2GRAY)

            # 데이터 프레임 형태로 변환             
            train=padded_face_gray.reshape(1,padded_face_gray.shape[0]*padded_face_gray.shape[1])

            df=pd.DataFrame(train)

            width, height, channel=182,182 ,1# 이미지 사이즈 182*182 pixel

            x_train=df.values
        
            x_train=x_train.reshape((x_train.shape [0],width,height ,channel))

            X=(x_train-127.5)/127.5 

            pred=self.model.predict(X)

            classes=['10대','20대','30대','40대','50대','60대','70대']
            
            return classes[np.argmax(pred)]
        except Exception as e:
            print('error:',e)
            err=['인식오류']
            return err  
