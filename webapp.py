from flask import Flask, request
import os
import torch
import easyocr
from ultralytics import YOLO
from controllers import predict_controller
import configparser
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')


# Get the root path of the project
root_path = os.path.dirname(os.path.abspath(__file__))

# get the yolov8 file
yolov8_file = config.get('YOLOV8_FILE_PATH', 'yolov8_file_path')
# get the yolov5 file
yolov5_file = config.get('YOLOV5_FILE_PATH', 'yolov5_file_path')

# Load in the yolov8 (license playes) and yolov5 (car brands) models
yolov8_model = YOLO(yolov8_file)
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_file, source='local')

# Apply confidence factor for YOLOv5
yolov5_model.conf = 0.7


# initialzing easyocr
reader = easyocr.Reader(['en'] , gpu = True)

# Create an instance of the PredictController
predict_controller = predict_controller.PredictController(yolov8_model, yolov5_model, reader, root_path)

# Register the endpoint by calling the controller method
@app.route('/predict', methods=['POST'])
def predict():
    return predict_controller.predict(request)

if __name__ == '__main__':
    app.run()

    








