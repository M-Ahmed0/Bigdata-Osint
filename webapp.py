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


from ultralytics import YOLO

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
                f.save(filepath)

                image = cv2.imread(f)
                print("image ----------", image)
                if file_extension == 'jpg':
                    image_drawn = brand_predict(image)
                    image_drawn = license_predict(image_drawn)
                elif file_extension == 'mp4':
                    print("Videos are not supported yet, come back later")
        
    return render_template('index.html')

def brand_predict(image):
    results = yolov5_model(image)
    pred_boxes = results.xyxy[0].detach().numpy()
    image_drawn = image.copy()
    for box in pred_boxes:
        xmin, ymin, xmax, ymax, conf, cls = box
        print("xmin", xmin, "ymin",ymin, "xmax", xmax, "ymax", ymax) 
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # draw green rectangle


    return image_drawn

def license_predict(image):

    detections = yolo.predict(image)  
    license_plate,image_drawn = process_results(detections,image)

    app_token = 'vFOK1jHTLo7llP150mPktYWgJ'
    vehicle_data = get_vehicle_data(license_plate, app_token)
    # Save the drawn image as PNG ----- testing purposes
    output_path = 'test.png'
    cv2.imwrite(output_path, cv2.cvtColor(image_drawn, cv2.COLOR_RGB2BGR))
    return image

def get_vehicle_data(license_plate, app_token): 
    base_url = 'https://opendata.rdw.nl/resource/m9d7-ebf2.json' 
    params = { 'kenteken': license_plate.replace("-", ""), '$$app_token': app_token } 
    try: 
        response = requests.get(base_url, params=params) 
        response.raise_for_status() # Raise an exception for 4xx or 5xx errors 
        data = response.json() 
        print("license_plate",license_plate.replace("-", ""))
        print("json",data)
        return data 
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
                        
            #ocr
            result = reader.readtext(thresh)    
                
            text = ""
        
            for res in result:
                if len(result) == 1:
                    text = res[1]
        
                if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
                    text = res[1]

            print(text)
                   
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
            cv2.putText(image, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text

            return (text,image)



#---------------------------------------------------------------------------------------------------------------------------------------#
# # function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

          
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
#     return render_template('index.html', image_path=image_path)
#     #return "done"



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     model = torch.hub.load('.', 'custom','best.pt', source='local')
#     model.eval()
#     app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat



# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)


# @app.route('/', methods=['POST'])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
        
#             #getting the file name and storing in "f" variable
#             f = request.files['file']
#             # storing the uploaded file by the user in upload folder
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath, 'uploads', f.filename)
#             print("upload folder is ", filepath)
#             f.save(filepath)
            
#             global imgpath
#             predict_img.imgpath = f.filename
#             print("printing predict_img :::::: ", predict_img)

#             file_extension = f.filename.rsplit('.', 1)[1].lower()
              
            
#             if file_extension == 'jpg':
#                 img = cv2.imread(filepath)
#                 frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                
#                 image = Image.open(io.BytesIO(frame))
                
#                 # loading the model and perform the detection
#                 yolo = YOLO('best.pt')
#                 detections = yolo.predict(image) # we are also saving the file in the runs folder
#                 print("fdsfdsg",detections)
#                 # here maybe add function to read the license plate using OCR 
                
#                 return display(f.filename)
                
                
#             elif file_extension == 'mp4':
#                 print("Videos are not supported yet, come back later")