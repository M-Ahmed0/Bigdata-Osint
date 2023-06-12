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

from flask_cors import CORS 
import base64
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return render_template('index.html')

# Load in the yolov8 (license playes) and yolov5 (car brands) models
yolov8_model = YOLO('./ml_models/lp_model.pt')
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./ml_models/brand_model.pt', source='local')

# Apply confidence factor for YOLOv5
yolov5_model.conf = 0.6

# Define app token for RDW API
app_token = 'vFOK1jHTLo7llP150mPktYWgJ'

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

    # save the image as a memory stream
    stream = BytesIO()
    stream.write(f.read())

    # Reset the stream position to the beginning
    stream.seek(0)

    validated_extension = validate_file_extension(file_extension)
    vehicle_data1=''
    if validated_extension == 'image':
        image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
        print("image ----------", image)
        if model=='BRAND':           
            image_drawn, vehicle_data = brand_predict(image)
        elif model == 'LP':
            image_drawn, vehicle_data = license_predict(image)
        elif model == 'BOTH':
            image_drawn, vehicle_data = brand_predict(image)
            image_drawn, vehicle_data = license_predict(image_drawn)
        else:
            return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
        
        # Saving the inference image temporarily
        output_image_path = 'output/image.jpg'
        cv2.imwrite(output_image_path, image_drawn)
        with open("output/image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data.__dict__ if not vehicle_data1 else vehicle_data1)}, 200
        # try: 
        #     return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data.__dict__ if not vehicle_data1 else vehicle_data1)}, 200
        # except: 
        #     return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data if not vehicle_data1 else vehicle_data1)}, 200
    elif validated_extension == 'video':

        if model=='BRAND':           
            img_arr = process_brand_video("test-video.mp4")
        elif model == 'LP':
            img_arr = process_lp_video("test-video.mp4")
        elif model == 'BOTH':
            img_arr = process_brand_video("test-video.mp4")
            img_arr_lp = process_lp_video("output/preprocess_video.mp4")
        else:
            return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
        
        construct_video_from_frames(img_arr)
        if model == 'BOTH':
            construct_video_from_frames(img_arr_lp)
        with open("output/preprocess_video.mp4", "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        return {'video': encoded_string.decode("utf-8")}, 200
    else:
        return jsonify({'error': 'Sorry, this extension is not supported.'}), 406

    # remove the image file from the server
    f.close()
    stream.close()

    # Return the inference image as the response
    #return jsonify(vehicle_data.__dict__ if not vehicle_data1 else vehicle_data1), 200
    #return jsonify({'image': image_drawn.tolist()}), 200
    # with open("output/image.jpg", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read())
    # try: 
    #     return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data.__dict__ if not vehicle_data1 else vehicle_data1)}, 200
    # except: 
    #     return {'image': encoded_string.decode("utf-8"), 'data': jsonstring(vehicle_data if not vehicle_data1 else vehicle_data1)}, 200
    # #return send_file(output_image_path, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run()    

def validate_file_extension(file_extension):
    image_extensions = ['jpeg', 'jpg', 'png']
    video_extensions = ['mp4', 'mov', 'avi']

    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"

def process_lp_video(video):
    detections = yolov8_model.predict(video)
    processed_frames = process_lp_video_frames(detections)
    return processed_frames
    
# start of brand related code
def brand_predict(image):
    results = yolov5_model(image)
    pred_boxes = results.xyxy[0].detach().numpy()

    class_name = ""
    
    #for class recognition
    class_names = results.names

    image_drawn = image.copy()
    for box in pred_boxes:
        xmin, ymin, xmax, ymax, conf, cls = box
        print("xmin", xmin, "ymin",ymin, "xmax", xmax, "ymax", ymax) 
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # class recognition
        class_name = class_names[int(cls)]
        print("Brand detected:", class_name)

        cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # draw green rectangle
        cv2.putText(image_drawn, class_name, (xmin-10, ymin-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text
    
    vehicle_dto = VehicleDTO.brand_dto(class_name)

    return image_drawn, vehicle_dto



def process_brand_video(video_path):
    video = cv2.VideoCapture(video_path)

    processed_frames = []

    while ret := video.grab():
        # Retrieve the grabbed frame
        ret, frame = video.retrieve()
        if not ret:
            break

        # Perform object detection and retrieve processed frame with bounding boxes and class names
        processed_frame, class_names = brand_predict(frame)
        
        processed_frames.append(processed_frame)

    # Release the video capture object
    video.release()
    print('Video processing completed')

    return processed_frames


def construct_video_from_frames(img_array):
    height, width, _ = img_array[0].shape
    size = (width, height)

    output_path = 'output/preprocess_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, size)

    for i in range(len(img_array)):
        out.write(img_array[i])


# end of brand related code

def license_predict(image):

    detections = yolov8_model.predict(image)  
    license_plate,image_drawn,vehicle_data = process_results(detections,image)

    # Save the drawn image as PNG ----- testing purposes
    output_path = 'test.png'
    cv2.imwrite(output_path, cv2.cvtColor(image_drawn, cv2.COLOR_RGB2BGR))
    return (image,vehicle_data)

def get_vehicle_data(license_plate, app_token): 
    base_url = 'https://opendata.rdw.nl/resource/m9d7-ebf2.json' 
    params = { 'kenteken': license_plate, '$$app_token': app_token } 
    try: 
        response = requests.get(base_url, params=params) 
        response.raise_for_status() # Raise an exception for 4xx or 5xx errors 
        data = response.json() 
        if len(data) ==0:
            return data
        vehicle_dto = VehicleDTO.from_json(data)
        return vehicle_dto 
    
    except requests.exceptions.RequestException as e: 
        print(f"An error occurred: {e}")


# initialzing easyocr
reader = easyocr.Reader(['en'] , gpu = True)

# ---- process all the results by enhancing the pixels and performing OCR to read the license plates.
def process_results(results, image):
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
                
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            
            x,y,w,h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            img = r.orig_img[y:h , x:w]
                            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
            
            # Apply threshold
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                           
                
            text = read_text_ocr(thresh)

            vehicle_data = get_vehicle_data(text, app_token)
            if vehicle_data=="":
                # Create a kernel for dilation and erosion
                kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
                # Apply dilation
                dilated_image = cv2.dilate(img, kernel, iterations=1)
                # Apply erosion
                eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
                #ocr
                text = read_text_ocr(eroded_image)
                vehicle_data = get_vehicle_data(text, app_token)

            print(text)
                   
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
            cv2.putText(image, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text

            return (text,image,vehicle_data)

# ---- process all the results by enhancing the pixels and performing OCR to read the license plates.
def process_lp_video_frames(results):
    processed_frames = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
                
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            
            x,y,w,h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            img = r.orig_img[y:h , x:w]
                            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
            
            # Apply threshold
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                           
                
            text = read_text_ocr(thresh)
            print(text)
                   
            cv2.rectangle(r.orig_img, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
            cv2.putText(r.orig_img, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text
            processed_frames.append(r.orig_img)

            #return (text,r.orig_img)
    
    return processed_frames


def clean_license_plate(license_plate): 
    cleaned_plate = re.sub(r"[^a-zA-Z0-9]", "", license_plate)
    return cleaned_plate.upper()

def read_text_ocr(img):
    #ocr
    result = reader.readtext(img)                   
    text = ""
        
    for res in result:
        if len(result) == 1:
            text = res[1]
        
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
    text = clean_license_plate(text)
    return text
#---------------------------------------------------------------------------------------------------------------------------------------#