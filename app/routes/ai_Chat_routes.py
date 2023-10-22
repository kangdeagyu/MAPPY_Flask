from flask_restx import Resource, abort
from flask import jsonify, request
from ..services.ai_service import AI_Service
import warnings


def aiChat_routes(aiChat_ns):
    @aiChat_ns.route("/chat")
    class AiTest(Resource):
        @aiChat_ns.doc(responses={
            200: '성공',
            300: 'Legacy',
            400: "Bad request. need 'new_sentence'- 잘못된 요청",
            500: "Cannot find the AI Model- 서버오류"
        })
        def post(self):          
            try:
                warnings.filterwarnings('ignore')
                
            except OSError:
                abort(500, error="Cannot find the AI Model")
                
            return jsonify({'result': "d"})
