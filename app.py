import os
from flask import Flask, render_template, request, jsonify
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'DenseNet.h5'
model = load_model(MODEL_PATH)

# Load the class indices
class_indices = json.load(open('class_indices_densenet.json', 'r'))

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        if 'image' not in request.files:
            return "No file uploaded"
        
        f = request.files['image']
        
        # Check if the file is selected and has a valid filename
        if f.filename == '':
            return "No file selected"
        
        # Generate a unique filename or use a timestamp
        filename = secure_filename(f.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        # Specify the directory to save the file
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process the result
        pred_class = np.argmax(preds, axis=1)
        result = class_indices[str(pred_class[0])]
        
        
        df = pd.read_csv('plant_diseases.csv')

        # Find the row in the CSV file that matches the result
        row = df[df['Folder Name'] == result]

        # If a matching row is found, get the plant and disease
        if not row.empty:
            plant = row['Plant Name'].values[0]
            disease = row['Disease'].values[0]
        else:
            plant = "Not available"
            disease = "Not available"
            
        return render_template('result.html', plant=plant, disease=disease)
    return None

if __name__ == '__main__':
    app.run(debug=True)
