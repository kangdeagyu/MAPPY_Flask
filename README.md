# Python 딥러닝 모델 Flask 서버 입니다

Python으로 구현된 Flask 어플리케이션 템플릿입니다.

## 🖇️ 준비 및 확인사항

### 패키지 명세
- 빌드 시 어플리케이션에 사용된 패키지를 설치하기 위해서는 `requirements.txt` 파일이 반드시 필요합니다.

## 기능 설명
- 사용자로부터 얼굴 이미지를 받아 OpenCV와 Dlib 라이브러리를 활용하여 얼굴의 형태를 추출합니다. 이 추출된 얼굴 형태 데이터는 학습된 Python 모델에 입력되어, 사용자의 얼굴 나이를 예측합니다. 이렇게 예측된 결과는 JSON 형태로 앱에 전송되어, 사용자에게 제공됩니다.
- 사용자로부터 입력받은 텍스트는 적절한 전처리 과정을 거친 후, 학습된 딥러닝 모델에 입력됩니다. 이 모델은 텍스트 데이터를 바탕으로 사용자의 의도를 예측하며, 이 예측 결과는 마찬가지로 앱에 전송됩니다. 이를 통해 사용자는 AI 챗봇과 원활하게 대화를 이어나갈 수 있습니다.
