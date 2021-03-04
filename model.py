from flask import Flask
from marshmallow import Schema, fields, pre_load, validate
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy


ma = Marshmallow()
db = SQLAlchemy()


class Users(db.Model):
    __tablename__ = 'users'
    __table_args__ = tuple(db.UniqueConstraint('id', 'users', name='my_name'))

    id = db.Column(db.String(), primary_key=True, unique=True)
    api_key = db.Column(db.String(), primary_key=True, unique=True)
    username = db.Column(db.String())
    first_name = db.Column(db.String())
    last_name = db.Column(db.String())
    password = db.Column(db.String())
    email = db.Column(db.String())

    def __init__(self, id, api_key, username, first_name, last_name, password, email):
        self.id = id
        self.username = username
        self.api_key = api_key
        self.first_name = first_name
        self.last_name = last_name
        self.password = password
        self.email = email

    def __repr__(self):
        return '<id {}>'.format(self.id)

    def serialize(self):
        return{
            'id': self.id,
            'api_key': self.api_key,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'password': self.password,
            'email': self.email,
        }


class Images(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    images = db.Column(db.String(150), unique=True, nullable=False)

    def __init__(self, images):
        self.images = images


class Comment(db.Model):
    __tablename__ = 'comments'
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(250), nullable=False)
    creation_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id', ondelete='CASCADE'), nullable=False)
    category = db.relationship('Category', backref=db.backref('comments', lazy='dynamic' ))

    def __init__(self, comment, category_id):
        self.comment = comment
        self.category_id = category_id


class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)

    def __init__(self, name):
        self.name = name


class CategorySchema(ma.Schema):
    id = fields.Integer()
    name = fields.String(required=True)


class CommentSchema(ma.Schema):
    id = fields.Integer(dump_only=True)
    category_id = fields.Integer(required=True)
    comment = fields.String(required=True, validate=validate.Length(1))
    creation_date = fields.DateTime()