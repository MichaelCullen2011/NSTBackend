import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import nst_lite as NST

root_dir = os.path.dirname(os.path.abspath(__file__))   # os.getcwd()
UPLOAD_FOLDER = root_dir + '/images/generated/lite/'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FLUTTER_JSON'] = {}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # 16 MB file size limit


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return 'Hello'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploaded/<filename>')
def uploaded_file(filename):
    os.listdir('images/generated/lite')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/nst',  methods=['POST', 'GET'])
def nst():
    if request.method == 'POST':
        content = request.json.get('content')
        style = request.json.get('style')
        data = {}
        filename = content + '-' + style + '.jpg'

        if not check_exists(filename):
            data['path'], image = NST.nst_csv(content, style)
            data['url'] = 'http://192.168.0.14:5000/uploaded/' + '{}-{}.jpg'.format(content, style)
            print(os.listdir('images/generated/lite'))
            return jsonify(data)
        else:
            data['path'] = os.getcwd() + '/images/generated/lite/' + '{}-{}.jpg'.format(content, style)
            data['url'] = 'http://192.168.0.14:5000/uploaded/' + '{}-{}.jpg'.format(content, style)
            return jsonify(data)

    elif request.method == 'GET':
        content = request.json.get('content')
        style = request.json.get('style')
        filename = content + '-' + style + '.jpg'
        if check_exists(filename):
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        else:
            return 'No file to get'


def check_exists(filename):
    for item in os.listdir('images/generated/lite'):
        if item == filename:
            return True
        else:
            return False


if __name__ == '__main__':
    app.run()
