from flask_restful import Resource


class Images(Resource):
    def get(self):
        return {"image": "Dog1.jpg"}






