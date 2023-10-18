from flask_restx import Resource, fields, abort, reqparse
from flask import jsonify, request
from werkzeug.datastructures import FileStorage
from ..services.ai_service import AI_Service
import warnings

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

def ai_routes(ai_ns):
    @ai_ns.route("/faceAge")
    class AiTest(Resource):
        @ai_ns.expect(upload_parser)
        # 오류 커맨드
        @ai_ns.doc(responses={
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
                ai_service = AI_Service()
                faceAge = ai_service.AI_predict(image_file)   
                print(faceAge)
            except OSError:
                abort(500, error="Cannot find the AI Model")
                
            return jsonify({'result': faceAge})
