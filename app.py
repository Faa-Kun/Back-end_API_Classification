import tensorflow as tf
from flask import Flask,  request, jsonify
from flask_cors import CORS
from keras.models import load_model
from tensorflow import expand_dims
from keras.applications.xception import preprocess_input, decode_predictions
from keras_preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename





app = Flask(__name__)
CORS(app)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = './'


def load_model_xception():
    model = load_model('model_batik_method.h5')
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        fail = request.files['image']
        if fail and allowed_file(fail.filename):
            
            failupload =  secure_filename(fail.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], failupload)
            fail.save(file_path)
            preds = process_predict(file_path) 
            result = str(preds)
            res = [{
                'Cetak' : str(preds[0]),
                'Tulis' : str(preds[1])
            }]
            return jsonify(res)

def process_predict(img_path):
    img_load = image.load_img(img_path, target_size=(150, 150))
    arr_img = image.img_to_array(img_load)
    arr_img = expand_dims(arr_img, axis = 0)
    x =  preprocess_input(arr_img)  
    
    img = np.vstack([x])
    model = load_model_xception()
    predict = model.predict(img)[0]  
    return predict
    
    



if __name__ == "__main__":
    app.run(debug = True)
