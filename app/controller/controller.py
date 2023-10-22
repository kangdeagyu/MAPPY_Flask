from flask_restx import Namespace

from ..routes.ai_faceAge_routes import aiFace_routes
from ..routes.ai_Chat_routes import aiChat_routes


def register_namespaces(api):
    aiFace_ns = Namespace("FaceModel")
    aiChat_ns = Namespace("ChatModel")

    aiFace_routes(aiFace_ns)
    aiChat_routes(aiChat_ns)

    api.add_namespace(aiFace_ns)
    api.add_namespace(aiChat_ns)