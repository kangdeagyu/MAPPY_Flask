#-*- coding: utf-8 -*-

from flask import Flask
from flask_restx import Api
import warnings
from .controller.controller import register_namespaces

app = Flask(__name__)
api = Api()
register_namespaces(api)
api.init_app(app)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app.run(host="127.0.0.1", port=5000, debug=True)