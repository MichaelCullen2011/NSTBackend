from flask import Blueprint
from flask_restful import Api
from resources.user import Users
from resources.images import Images
from resources.category import CategoryResource
from resources.comment import CommentResource

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

# Route
api.add_resource(Users, '/users')
api.add_resource(Images, '/images')
api.add_resource(CategoryResource, '/category')
api.add_resource(CommentResource, '/comment')

'''
Using /Category:
    
    # Post
    db.session.add(category)
    db.session.commit()
    result = category_schema.dump(category).data
    return { "status": 'success', 'data': result }, 201
    
    # Put
    result = category_schema.dump(category).data
    return { "status": 'success', 'data': result }, 204
    
    # Delete
    result = category_schema.dump(category).data
    return { "status": 'success', 'data': result}, 204
'''




