from flask_restx import Resource, abort, reqparse
from flask import jsonify, request
from ..services.ai_chat_service import AI_ChatService
import warnings

# Create a parser instance.
parser = reqparse.RequestParser()
parser.add_argument('text', type=str, help='The text to analyze', required=True)

def aiChat_routes(aiChat_ns):
    @aiChat_ns.route("/chat")
    class AiTest(Resource):
        @aiChat_ns.expect(parser)
        @aiChat_ns.doc(responses={
            200: '성공',
            300: 'Legacy',
            400: "Bad request. need 'new_sentence'- 잘못된 요청",
            500: "Cannot find the AI Model- 서버오류"
        })
        def post(self):
            
            args = parser.parse_args()
            text = args['text']
            try:
                warnings.filterwarnings('ignore')
                ai_service = AI_ChatService()
                chat = ai_service.evaluate(text)
            except OSError:
                abort(500, error="Cannot find the AI Model")
                
            return jsonify({'result': chat})
