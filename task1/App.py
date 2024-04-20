import os
import argparse
from flask import Flask, flash, request, redirect, render_template,jsonify, send_file,send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

from tools.prepare_data import main_prepare
from tools.label_reformulate import main_reformat
from tools.cropping_stuff import get_body_mask, get_bounding_box, get_cropped_volumes, crop_to_fullres
import tempfile
import shutil

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
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['dcm', 'nii.gz', 'dicom', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/<path:path>')
def home(path):
  return render_template(path)


@app.route('/predict', methods=['POST'])
def process_data():
    
    base_directory = './input'
    nnunet_in = tempfile.mkdtemp(dir=base_directory)
    nnunet_out = tempfile.mkdtemp(dir=base_directory)
    # main_path_in = os.path.join(temp_directory, 'main_path_in')
    # save_path_out = os.path.join(temp_directory, 'save_path_out')
    
    # body_mask_path = os.path.join(main_path_in, 'cropped_in')
    # crop_log_path = os.path.join(main_path_in, 'crop_log')
    # fullres_in = os.path.join(main_path_in, 'model_in')
    # nnunet_in = os.path.join(main_path_in, 'nnunet_in')
    # nnunet_out = os.path.join(save_path_out, 'seg_cropped_int')
    # fullsize_seg_path = os.path.join(save_path_out ,'seg_fullres_int')
    # print('\n'*5)
    # print("before main_prepare")
    # # main_prepare(main_path_in, save_path_out)
    # print('\n'*5)
    # print("After main_prepare")
    files = request.files['files[]']
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(nnunet_in.name +"/" +filename))
            else:
                print("Exception occurred during saving the file..")
            
    except Exception as e:
        print("Exception occurred:", str(e))
        
   
    
    # get_body_mask(fullres_in, body_mask_path)
    
    # get_bounding_box(body_mask_path)
    
    # get_cropped_volumes(fullres_in, body_mask_path, nnunet_in, crop_log_path)
    
    os.system('nnUNet_predict -i %s -o %s -t 606 -m 3d_fullres -tr nnUNetTrainerV2_noMirroring -f=all --disable_tta  --mode fast' % (nnunet_in.name, nnunet_out.name))
    
    # crop_to_fullres(save_path_out, crop_log_path, fullsize_seg_path)
    
    # main_reformat(save_path_out)
    
    # # Return the segmented processed Nifti file
    # segmented_file = os.path.join(save_path_out, 'segmented.nii.gz')
    # shutil.move(fullsize_seg_path, segmented_file)
    # return send_file(segmented_file, mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")
       
    # return segmented_file
    files = os.listdir(nnunet_out.name)
    retFile = files[0]
    return send_file(nnunet_out.name +"/"+retFile, mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")
       

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)
