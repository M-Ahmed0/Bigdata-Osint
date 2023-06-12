import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response , jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from flask import jsonify
import matplotlib.pyplot as plt
import easyocr
from io import BytesIO
from collections import Counter
from json import dumps as jsonstring
from ultralytics import YOLO
from VehicleDTO import VehicleDTO
from classes.brand_service import Brand_service
from classes.license_plate_service import license_plate_service

from flask_cors import CORS 
import base64
app = Flask(__name__)
CORS(app)

# Load in the yolov8 (license playes) and yolov5 (car brands) models
yolov8_model = YOLO('./ml_models/lp_model.pt')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./ml_models/brand_model.pt', source='local')

# Apply confidence factor for YOLOv5
yolov5_model.conf = 0.6



# initialzing easyocr
reader = easyocr.Reader(['en'] , gpu = True)

# predict end point
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    model = request.args.get("model")    
    print(model)
    f = request.files['file']
    # storing the uploaded file by the user in upload folder
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', f.filename)
    file_extension = f.filename.rsplit('.', 1)[1].lower()
    print("upload folder is ", filepath)

    f.save(filepath)

    validated_extension = validate_file_extension(file_extension)
    

    brand = Brand_service()
    license_plate=license_plate_service()

    if validated_extension == 'image':
        image = cv2.imread(filepath)
        if model=='BRAND':
            image_drawn, vehicle_data = brand.brand_predict(image, yolov5_model)           
        elif model == 'LP':
            image_drawn, vehicle_data = license_plate.license_predict(image,yolov8_model,reader)
        elif model == 'BOTH':
            image_drawn, vehicle_data = brand.brand_predict(image, yolov5_model)
            image_drawn, vehicle_data = license_plate.license_predict(image_drawn,yolov8_model,reader)
        else:
            return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
        
        # Saving the inference image temporarily
        output_image_path = 'output/image.jpg'
        cv2.imwrite(output_image_path, image_drawn)
        with open("output/image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        # remove the image file from the server
        f.close()
        os.remove(filepath)
        return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data.__dict__), 'type': 'image'}, 200
    
    elif validated_extension == 'video':
        if model=='BRAND':           
            img_arr = brand.process_brand_video(filepath,yolov8_model)
        elif model == 'LP':
            img_arr = license_plate.process_lp_video(filepath,yolov8_model,reader)
        elif model == 'BOTH':
            img_arr = brand.process_brand_video(filepath,yolov8_model)
            
        else:
            return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
        
        construct_video_from_frames(img_arr)
        if model == 'BOTH':
            img_arr_lp = license_plate.process_lp_video("output/preprocess_video.mp4",yolov8_model)
            construct_video_from_frames(img_arr_lp)
        with open("output/preprocess_video.mp4", "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        # remove the image file from the server
        test = encoded_string.decode("utf-8")
        os.remove(filepath)
        return {'video': test, 'type': 'video', 'data': ''}, 200
    else:
        return jsonify({'error': 'Sorry, this extension is not supported.'}), 406




if __name__ == "__main__":
    app.run()    

def validate_file_extension(file_extension):
    image_extensions = ['jpeg', 'jpg', 'png']
    video_extensions = ['mp4', 'mov', 'avi']

    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"


def construct_video_from_frames(img_array):
    height, width, _ = img_array[0].shape
    size = (width, height)

    output_path = 'output/preprocess_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, size)

    for i in range(len(img_array)):
        out.write(img_array[i])






