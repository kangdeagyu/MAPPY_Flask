from flask_restx import Resource, abort, reqparse
from flask import jsonify, request
from werkzeug.datastructures import FileStorage
from ..services.ai_face_service import AI_FaceService
import warnings

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
                faceAge = ai_service.AI_predict(image_file)   
                print(faceAge)
            except OSError:
                abort(500, error="Cannot find the AI Model")
                
            return jsonify({'result': faceAge})
