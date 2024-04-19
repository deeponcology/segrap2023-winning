from mailbox import Message
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import subprocess
import json
import shutil
import tempfile
from flask import jsonify, send_file
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = "./input"
# os.path.join(path, 'uploads')

# Make directory if uploads is not exists
# if not os.path.isdir(UPLOAD_FOLDER):
#     os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['dcm', 'nii.gz', 'dicom', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/<path:path>')
def home(path):
  return render_template(path)

@app.route('/predict/<path:path>', methods=['POST'])
def predict(path):

    print("inside ---",path)
    
    # os.mkdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':

        
        print("inside ---")
        files = request.files.getlist('files[]')
        inputDir = tempfile.TemporaryDirectory(dir="./input")
        
        outDir = tempfile.TemporaryDirectory(dir="./output")
        print(inputDir.name)
        print(outDir.name)

        for file in files:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(inputDir.name +"/" +filename)
            

       
            my = os.listdir(app.config['UPLOAD_FOLDER'])
            print("input dir = ",my)
            # nnUNet_predict -i $inputDir -o $outDir --task_name $1 --model 2d --disable_tta
        # 
            # subprocess.check_output("/home/predict.sh", shell=True)
            subprocess.check_output(
                [
                "nnUNet_predict", 
                "-i", inputDir.name,
                "-o", outDir.name,
                "-t", path,
                "-m", "2d",
                "--disable_tta"]
                )
        files = os.listdir(outDir.name)
        retFile = files[0]
        return send_file(outDir.name +"/"+retFile, mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")
       
 
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
