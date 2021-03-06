import os
import base64
import json
from io import BytesIO
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import nst_lite as NST

root_dir = os.getcwd()
UPLOAD_FOLDER = root_dir + '\\images\\generated\\lite\\'
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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/run/<content>-<style>')
def run_nst(content, style):
    b = BytesIO()
    data = {}

    data['path'], image = NST.nst_csv()
    image.save(b, format="jpeg")
    b.seek(0)

    data['image'] = base64.b64encode(b.read()).decode('ascii')
    data['url'] = '/uploads/' + '{}-{}'.format(content, style)
    return json.dumps(data)


@app.route('/app/<content>-<style>', methods=['POST', 'GET'])
def edit_csv(content, style):
    if request.method == 'POST':
        return redirect(url_for('run_nst', content=content, style=style))
    elif request.method == 'GET':
        return redirect(url_for('uploaded_file', content=content, style=style))

    return app.config['FLUTTER_JSON']


if __name__ == '__main__':
    app.run()
