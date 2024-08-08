from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from src.yolov8_model import load_yolov8, predict_yolov8, process_results
import os

app = Flask(__name__)

# Load the YOLOv8 model
model = load_yolov8('s')  # You can change 's' to other sizes if needed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Read the image
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            # Perform prediction
            results = predict_yolov8(model, image)
            
            # Process results
            output_image = process_results(results, image)
            
            # Encode the result image
            _, buffer = cv2.imencode('.jpg', output_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({'image': encoded_image})
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
