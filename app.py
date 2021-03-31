import os
from PIL import Image
import numpy as np
import io
import base64
import json
from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import nst_lite
import nst_detail

root_dir = os.path.dirname(os.path.abspath(__file__))   # os.getcwd()
UPLOAD_FOLDER = root_dir + '/images/generated/lite/'
DETAIL_FOLDER = root_dir + '/images/generated/detail/'
CONTENT_FOLDER = root_dir + '/images/content/'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
# app.config['SECRET_KEY'] = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
app.config['UPLOAD_FOLDER'] = DETAIL_FOLDER
app.config['LITE_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['FLUTTER_JSON'] = {}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # 16 MB file size limit


@app.route('/')
def home():
    return 'Hello'


@app.route('/uploaded/<filename>', methods=['POST', 'GET'])
def uploaded_file(filename):
    if request.method == 'GET':
        print("Displaying Generated File {}...".format(filename))
        os.listdir(app.config['UPLOAD_FOLDER'])
        network_image = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        return network_image


@app.route('/nst',  methods=['POST', 'GET'])
def nst():
    if request.method == 'POST':
        print("Request Received...")
        delete_files()
        # Uploading Image
        uploaded = False
        file = request.files['file']
        data = json.loads(request.form.get('data'))
        lite_vers = data["lite"]
        if lite_vers == 'true':
            lite = True
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        elif lite_vers == 'false':
            lite = True     # CURRENTLY THE DETAILED VERSION TAKES TOO LONG TO GENERATE
            app.config['UPLOAD_FOLDER'] = DETAIL_FOLDER
        else:
            lite = True
            app.config['UPLOAD_FOLDER'] = DETAIL_FOLDER

        if file:
            filename = secure_filename(file.filename)
            exists = check_exists(filename)
            if not exists:
                file.save(os.path.join(app.config['CONTENT_FOLDER'], filename))
                rotate_image(os.path.join(app.config['CONTENT_FOLDER'], filename))
                uploaded = True
                os.listdir(app.config['CONTENT_FOLDER'])

        # Generating New Image
        print("Grabbing data from request...")
        content = data["content"]
        style = data["style"]
        data = {}
        filename = content + '-' + style + '.jpg'

        if not check_generated(filename):
            if lite:
                data['path'], image = nst_lite.run(content, style)
                data['url'] = request.host_url + '/uploaded/' + '{}-{}.jpg'.format(content, style)
            elif not lite:
                data['path'], image = nst_detail.run(content, style)
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
    print("Checking if {} has been generated already...".format(filename))
    for item in os.listdir(app.config['LITE_FOLDER']):
        print(item, filename)
        if item == filename:
            print("Generated!")
            return True
    else:
        print("Not Generated!")
        return False


def check_exists(filename):
    print("Checking if {} exists in content folder...".format(filename))
    for item in os.listdir(app.config['CONTENT_FOLDER']):
        if item == filename:
            print("Exists!")
            return True
    else:
        print("Does Not Exist!")
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
        print('Uploaded Camera File Deleted')
    else:
        print("Uploaded File NOT Camera File")


def delete_files():
    print("Deleting generated files...")
    for file in os.listdir(app.config['LITE_FOLDER']):
        print(file)
        if file == 'Dog1-Kandinsky.jpg':
            pass
        else:
            os.remove(app.config['LITE_FOLDER'] + file)
            print("Deleted: ", file)


def rotate_image(image_path):
    print("Orienting Camera Image...")
    unrotated = Image.open(image_path)
    rotated = unrotated.rotate(270)
    rotated.save(image_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
