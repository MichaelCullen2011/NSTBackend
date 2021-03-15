import os
from PIL import Image
import numpy as np
import io
import base64
import json
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import nst_lite as NST

root_dir = os.path.dirname(os.path.abspath(__file__))   # os.getcwd()
UPLOAD_FOLDER = root_dir + '/images/generated/lite/'
CONTENT_FOLDER = root_dir + '/images/content/'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
# app.config['SECRET_KEY'] = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['FLUTTER_JSON'] = {}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # 16 MB file size limit


@app.route('/')
def home():
    return 'Hello'


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            exists = check_exists(filename)
            if not exists:
                print("UPLOAD FILENAME", filename)
                file.save(os.path.join(app.config['CONTENT_FOLDER'], filename))
                return "SAVED"
            else:
                print("Image Exists")
                return "EXISTS"


@app.route('/uploaded/<filename>', methods=['POST', 'GET'])
def uploaded_file(filename):
    if request.method == 'GET':
        os.listdir(app.config['UPLOAD_FOLDER'])
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/nst',  methods=['POST', 'GET'])
def nst():
    if request.method == 'POST':
        # Uploading Image
        print("Grabbing file from request")
        uploaded = False
        file = request.files['file']
        if file:
            print("Checking if the file exists...")
            filename = secure_filename(file.filename)
            exists = check_exists(filename)
            if not exists:
                print("UPLOAD FILENAME", filename)
                file.save(os.path.join(app.config['CONTENT_FOLDER'], filename))
                uploaded = True
                os.listdir(app.config['CONTENT_FOLDER'])
            else:
                print("Image Exists")

        # Generating New Image
        print("Grabbing data from request")
        data = json.loads(request.form.get('data'))
        content = data["content"]
        style = data["style"]
        data = {}
        filename = content + '-' + style + '.jpg'

        if not check_generated(filename):
            data['path'], image = NST.nst_csv(content, style)
            data['url'] = request.host_url + '/uploaded/' + '{}-{}.jpg'.format(content, style)
            print(data)
            if uploaded:
                delete_camera_image(content)
            return jsonify(data)
        else:
            data['path'] = app.config['UPLOAD_FOLDER'] + '{}-{}.jpg'.format(content, style)
            data['url'] = request.host_url + '/uploaded/' + '{}-{}.jpg'.format(content, style)
            print(data)
            if uploaded:
                delete_camera_image(content)
            return jsonify(data)

    elif request.method == 'GET':
        content = request.json.get('content')
        style = request.json.get('style')
        filename = content + '-' + style + '.jpg'
        if check_generated(filename):
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        else:
            return 'No file to get'


def check_generated(filename):
    print("CHECK GENERATED", filename)
    for item in os.listdir(app.config['UPLOAD_FOLDER']):
        if item == filename:
            return True
    else:
        return False


def check_exists(filename):
    print("CHECK EXISTS ", filename)
    for item in os.listdir(app.config['CONTENT_FOLDER']):
        if item == filename:
            print("EXISTS")
            return True
    else:
        print("DOESNT EXIST")
        return False


def convert_to_image():
    file = request.files['image'].read()  # byte file
    npimg = np.fromstring(file, np.uint8)
    img = Image.fromarray(npimg)
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "JPEG")
    raw_bytes.seek(0)
    img_base64 = base64.b64encode(raw_bytes.read())
    return jsonify({'status': str(img_base64)})


def delete_camera_image(content):
    if content == 'camera_image':
        os.remove((os.path.join(app.config['CONTENT_FOLDER'], content + '.jpg')))
        print('Uploaded File Deleted')
    else:
        print("Uploaded File NOT Deleted")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
