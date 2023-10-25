import json
from flask_restx import Resource, abort, reqparse
from io import BytesIO
from flask import jsonify, request, send_file
from werkzeug.datastructures import FileStorage
from ..services.ai_face_service import AI_FaceService
import warnings
from PIL import Image

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

def aiFace_routes(aiFace_ns):
    @aiFace_ns.route("/faceAge")
    class AiTest(Resource):
        @aiFace_ns.expect(upload_parser)
        # 오류 커맨드
        @aiFace_ns.doc(responses={
            200: '성공',
            300: 'Legacy',
            400: "Bad request. need 'new_sentence'- 잘못된 요청",
            500: "Cannot find the AI Model- 서버오류"
        })
        def post(self):          
            if 'file' not in request.files:
                abort(400, error="No file part")
            
            image_file = request.files['file']
            
            try:
                warnings.filterwarnings('ignore')
                ai_service = AI_FaceService()
                faceAge, percent = ai_service.AI_predict(image_file)  
                flat_data = [item for sublist in percent for item in sublist]
                
                print(">>>", faceAge)
                print(">>>", flat_data[0])
            except OSError:
                abort(500, error="Cannot find the AI Model")
                
            return jsonify({'age': faceAge, 'percent' : eval(str(flat_data))})
        
    @aiFace_ns.route("/faceCrop")
    class FaceCrop(Resource):
        @aiFace_ns.expect(upload_parser)
        @aiFace_ns.doc(responses={
            200: '성공',
            300: 'Legacy',
            400: "Bad request. need 'new_sentence'- 잘못된 요청",
            500: "Cannot find the AI Model- 서버오류"
        })
        def post(self):
            if 'file' not in request.files:
                abort(400, error="No file part")

            image_file = request.files['file']

            try:
                warnings.filterwarnings('ignore')
                ai_service = AI_FaceService()
                _, _ = ai_service.AI_predict(image_file)  # 얼굴 인식 수행
                cropped_image = ai_service.cropped_image  # 잘려진 이미지 가져오기
                cropped_image_pil = Image.fromarray(cropped_image)  # Numpy 배열을 PIL 이미지로 변환
                cropped_image_bytes = BytesIO()  # 이미지를 바이트 스트림으로 저장하기 위한 BytesIO 객체 생성
                cropped_image_pil.save(cropped_image_bytes, format='JPEG')  # 이미지를 JPEG 형식으로 저장
                cropped_image_bytes.seek(0)  # 바이트 스트림의 포인터 위치를 처음으로 이동

                return send_file(cropped_image_bytes, mimetype='image/jpeg')  # 잘려진 이미지 반환

            except OSError:
                abort(500, error="서버 오류")
