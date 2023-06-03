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

from ultralytics import YOLO
from VehicleDTO import VehicleDTO
app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')

 # loading yolov8 model
yolo = YOLO('LP_model_best.pt')

#loading yolov5 model (device prob not needed, but ill leave it here for now)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='200_epoch_adam_brand_detection.pt', force_reload=True)
yolov5_model = yolov5_model.to(device)
app_token = 'vFOK1jHTLo7llP150mPktYWgJ'

@app.route('/', methods=['POST', 'GET'])
def predict_with_yolov5():
    if request.method == "POST":
        if 'file' in request.files:
                #getting the file name and storing in "f" variable
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

                image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
                print("image ----------", image)

                validated_extension = validate_file_extension(file_extension)

                if validated_extension == 'image':
                    image_drawn, bounding_boxes = brand_predict(image)
                    image_drawn = license_predict(image_drawn)
                elif validated_extension == 'video':
                    test = process_brand_video("test.mp4")
                    print(test)
                else:
                    print("Sorry, this extension is not supported.")

                # remove the image file from the server
                f.close()
                stream.close()

                
        
    return render_template('index.html')




# predict end point

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
        
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

    image = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
    print("image ----------", image)

    validated_extension = validate_file_extension(file_extension)

    if validated_extension == 'image':
        image_drawn, bounding_boxes = brand_predict(image)
        image_drawn = license_predict(image_drawn)
    elif validated_extension == 'video':
        test = process_brand_video("test.mp4")
        print(test)
    else:
        return jsonify({'error': 'Sorry, this extension is not supported.'}), 406

    # remove the image file from the server
    f.close()
    stream.close()
   
    # Saving the inference image temporarily
    output_image_path = 'output/image.jpg'
    cv2.imwrite(output_image_path, image_drawn)

    # Return the inference image as the response
    return send_file(output_image_path, mimetype='image/jpeg')






if __name__ == "__main__":
    app.run()    

def validate_file_extension(file_extension):
    image_extensions = ['jpeg', 'jpg', 'png']
    video_extensions = ['mp4', 'mov', 'avi']

    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"


def brand_predict(image):
    results = yolov5_model(image)
    pred_boxes = results.xyxy[0].detach().numpy()

    bounding_boxes = []

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

        bounding_boxes.append((xmin, ymin, xmax, ymax))

        cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # draw green rectangle


    return image_drawn, bounding_boxes

def process_brand_video(video_path):

    video = cv2.VideoCapture(video_path)
    processed_frames = []
    frame_count = 0

    # frames to skip (play around with this)
    process_interval = 5

    while ret := video.grab():
        # Increment the frame count
        frame_count += 1

        # Check if it's time to preprocess the frame
        if frame_count % process_interval != 0:
            continue

        # Retrieve the grabbed frame
        ret, frame = video.retrieve()
        if not ret:
            break

        # Perform object detection and retrieve processed frame and bounding box coordinates
        processed_frame, bounding_boxes = brand_predict(frame)
        processed_frames.append((processed_frame, bounding_boxes))

    # Release the video capture object
    video.release()
    print('Video processing completed')

    return processed_frames

def license_predict(image):

    detections = yolo.predict(image)  
    license_plate,image_drawn = process_results(detections,image)

    # Save the drawn image as PNG ----- testing purposes
    output_path = 'test.png'
    cv2.imwrite(output_path, cv2.cvtColor(image_drawn, cv2.COLOR_RGB2BGR))
    return image

def get_vehicle_data(license_plate, app_token): 
    base_url = 'https://opendata.rdw.nl/resource/m9d7-ebf2.json' 
    params = { 'kenteken': license_plate, '$$app_token': app_token } 
    try: 
        response = requests.get(base_url, params=params) 
        response.raise_for_status() # Raise an exception for 4xx or 5xx errors 
        data = response.json() 
        print(type(data))
        vehicle_dto = VehicleDTO.from_json(data)
        print("license_plate",license_plate)
        print("json",data)
        print(f"vehicle_dto: {vehicle_dto}")
        print(f"vehicle_dto: {vehicle_dto.license_plate}")
        print(f"vehicle_dto",vehicle_dto.brand)
        print(f"vehicle_dto",vehicle_dto.apk_expiry_date)
        print(f"vehicle_dto",vehicle_dto.veh_registration_nr)
        return vehicle_dto 
    except requests.exceptions.RequestException as e: 
        print(f"An error occurred: {e}")



@app.route('/yolov8', methods=['POST', 'GET'])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
        
            #getting the file name and storing in "f" variable
            f = request.files['file']
            # storing the uploaded file by the user in upload folder
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            

            file_extension = f.filename.rsplit('.', 1)[1].lower()
              
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                
                image = Image.open(io.BytesIO(frame))

                # model
                detections = yolo.predict(image)      
                # here maybe add function to read the license plate using OCR 
                print('YOLO v8 detections', detections)

                # return process_results(detections)
                
                # return display(f.filename)
                
                
            elif file_extension == 'mp4':
                print("Videos are not supported yet, come back later")
    
    return render_template('index.html')


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

            vehicle_data = get_vehicle_data(text.replace("-", ""), app_token)
            if vehicle_data=="":
                # Create a kernel for dilation and erosion
                kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
                # Apply dilation
                dilated_image = cv2.dilate(img, kernel, iterations=1)
                # Apply erosion
                eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
                #ocr
                text = read_text_ocr(eroded_image)

            print(text.replace("-", ""))
                   
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
            cv2.putText(image, text.replace("-", ""), (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text

            return (text,image)


def read_text_ocr(img):
    #ocr
    result = reader.readtext(img)                   
    text = ""
        
    for res in result:
        if len(result) == 1:
            text = res[1]
        
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
    return text
#---------------------------------------------------------------------------------------------------------------------------------------#