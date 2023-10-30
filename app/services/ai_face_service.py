import cv2
import dlib
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

class AI_FaceService:
    def __init__(self,
                    model_path='app/static/16-16_16_cnn_model.h5',
                    shape_predictor_path="app/static/shape_predictor_68_face_landmarks.dat",
                    target_size=(128, 128),
                    padding_size=(184, 184),
                    brightness_adjustment=83.15755552578428):
        self.model = load_model(model_path)
        # dlib 모델 불러오기
        self.detector = dlib.get_frontal_face_detector()
        self.target_size = target_size
        self.padding_size = padding_size
        self.brightness_adjustment = brightness_adjustment
        self.cropped_image = None; # 크롭이미지 전역변수로 선언

    

    def AI_predict(self, image_file):
        try:
            # 이미지 불러오기 (Image -> Numpy Array)
            img_pil = Image.open(image_file)
            img = np.array(img_pil)

            # BGR GRAY 색상 변환 (PIL은 RGB 순서이므로 RGB->BGR 변환 후 GRAY 색상 변환)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            faces = self.detector(img_bgr, 1)
            
            if len(faces) > 0:
                face = faces[0]

                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y

                imgCrop = img_bgr[y:y+h, x:x+w]
                self.cropped_image = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)

                # 잘린 이미지 사이즈 조절 (128,128)
                face_resized=cv2.resize(imgCrop,self.target_size)

                # 패딩 사이즈 설정
                padded_face=np.zeros((*self.padding_size,3),dtype=np.uint8)
                    
                x_offset=(self.padding_size[0]-self.target_size[0])//2
                y_offset=(self.padding_size[1]-self.target_size[1])//2

        except Exception as e:
            print('error in image processing:',e)
            return '인식오류'

        try:
            padded_face[y_offset:y_offset+self.target_size[1],x_offset:x_offset+self.target_size[0]]=face_resized

            # 학습된 이미지 평균 밝기로 밝기 조절
            adjusted_image = padded_face.astype(float) - np.mean(padded_face) + self.brightness_adjustment

            adjusted_image = np.clip(adjusted_image, 0, 255)   # 음수 값은 0으로 설정, 초과하는 값은 최댓값인 255로 설정

        except Exception as e:
            print('error in image adjustment:',e)
            return '인식오류'

        try:  
            img2 = Image.fromarray(adjusted_image.astype('uint8'))
            img2 = img2.convert('L')

            train=np.zeros(1*184*184,dtype=np.int32).reshape(1,184,184) 

            img=np.array(img2,dtype=np.int32)
            train[0,:,:]=img

            train=train.reshape(-1,184 * 184)

            df=pd.DataFrame(train)

        except Exception as e:
            print('error in converting image to grayscale:',e)
            return '인식오류'

        try: 
            width, height, channel=184 ,184 ,1 # 이미지 사이즈 184*184 pixel

            x_train=df.values
            
            x_train=x_train.reshape((x_train.shape [0],width,height ,channel))

            X=(x_train-127.5)/127.5 

            pred=self.model.predict(X)

        except Exception as e:
            print('error in model prediction:',e)
            return '인식오류'
            
        classes=['10대','20대','30대','40대','50대','60대','70대']
                
        return (classes[np.argmax(pred)],pred)
