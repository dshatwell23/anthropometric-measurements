from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug

import cv2
from nta import NTAModel

import os

app = Flask(__name__)
api = Api(app)

class UploadImage(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument("frontal", type=werkzeug.datastructures.FileStorage, location="files")
        parse.add_argument("lateral", type=werkzeug.datastructures.FileStorage, location="files")
        args = parse.parse_args()

        # Read images from buffer and save to memory
        image_file = args["frontal"]
        image_file.save("images/front.png")

        image_file = args["lateral"]
        image_file.save("lateral.png")

        # Image analysis
        frontal = cv2.rotate(cv2.imread("frontal.png"), cv2.ROTATE_90_CLOCKWISE)
        lateral = cv2.rotate(cv2.imread("lateral.png"), cv2.ROTATE_90_CLOCKWISE)

        nta = NTAModel(front, lateral)
        measurements = nta.find_distances()

        os.remove("frontal.png")
        os.remove("lateral.png")

        return measurements


api.add_resource(UploadImage, "/upload")

if __name__ == "__main__":
    app.run(debug=True, host= "0.0.0.0")