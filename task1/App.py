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
from tools.prepare_data import main_prepare
from tools.label_reformulate import main_reformat
from tools.cropping_stuff import get_body_mask, get_bounding_box, get_cropped_volumes, crop_to_fullres

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




@app.route('/<path:path>')
def home(path):
  return render_template(path)

@app.route('/predict', methods=['POST'])
def predict():
    main_path_in = tempfile.TemporaryDirectory(dir="./input")
    print("create main_path_in folder")
    save_path_out = tempfile.TemporaryDirectory(dir="./input")
    print("create save_path_out folder")
    main_path_in = os.path.join(main_path_in, 'images')
    save_path_out = os.path.join(save_path_out, 'images')
    # print("inside ---",path)
    
    # os.mkdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':

        
        print("inside ---")
        files = request.files.getlist('files[]')
       

        for file in files:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(main_path_in.name +"/" +filename)
            
            # nnUNet_predict -i $inputDir -o $outDir --task_name $1 --model 2d --disable_tta
        # 
            # subprocess.check_output("/home/predict.sh", shell=True)
            
        # body_mask_path = os.path.join(main_path_in.name, 'cropped_in')
        # crop_log_path = os.path.join(main_path_in.name, 'crop_log')
        # fullres_in = os.path.join(main_path_in.name, 'model_in')
        # nnunet_in = os.path.join(main_path_in.name, 'nnunet_in')
        # nnunet_out = os.path.join(save_path_out.name, 'seg_cropped_int')
        # fullsize_seg_path = os.path.join(save_path_out.name ,'seg_fullres_int')
            
        body_mask_path = main_path_in.name+ '/cropped_in'
        crop_log_path = main_path_in.name+  '/crop_log'
        fullres_in = main_path_in.name + '/model_in'
        nnunet_in = main_path_in.name + '/nnunet_in'
        nnunet_out = save_path_out.name + '/seg_cropped_int'
        fullsize_seg_path = save_path_out.name + '/seg_fullres_int'
        
        main_prepare(main_path_in.name, save_path_out)
        
        print('\n'*5)
        print(' Extracting body mask begins ...')
        print('\n'*5)    
        get_body_mask(fullres_in, body_mask_path)
        
        print('\n'*5)
        print('Calculating the BBox coordinates begins ...')
        print('\n'*5)   
        get_bounding_box(body_mask_path)
        
        print('\n'*5)
        print('Extracting cropped volume process begins ...')
        print('\n'*5)    
        get_cropped_volumes(fullres_in, body_mask_path, nnunet_in, crop_log_path)
        
        print('\n'*5)
        print('Segmenting the cropped volume begins ...')
        print('\n'*5)   
        os.system('nnUNet_predict -i %s -o %s -t 606 -m 3d_fullres -tr nnUNetTrainerV2_noMirroring -f=all --disable_tta  --mode fast' % (nnunet_in, nnunet_out))
        
        print('\n'*5)
        print('Projecting cropped mask into full resolutional masks begins ...')
        print('\n'*5)   
        crop_to_fullres(save_path_out, crop_log_path, fullsize_seg_path)    
        
        print('\n'*4)
        print('Segmentation process finished successfully!')
        print('\n'*4)
        main_reformat(save_path_out)
        
        print('\n'*4)
        print('The pipeline was executed successfully')
        files = os.listdir(save_path_out.name)
        retFile = files[0]
        return send_file(save_path_out.name +"/"+retFile, mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")
       
 
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
