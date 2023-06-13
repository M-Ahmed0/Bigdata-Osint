from json import dumps as jsonstring
import cv2
from flask import jsonify
from services.brand_service import BrandService
from services.license_plate_service import LicensePlateService
import os
import base64


class PredictController:
   
    # constructor for creation of controller instance and assignment of variables
    def __init__(self,  yolov8_model, yolov5_model, reader, root_path):
        
        self.yolov8_model = yolov8_model
        self.yolov5_model = yolov5_model
        self.reader = reader
        self.root_path = root_path
     
    # prediction end-point for running inference on images and videos
    def predict(self, request):

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        model = request.args.get("model")    
        print(model)
        f = request.files['file']
        
        filepath = os.path.join(self.root_path, 'uploads', f.filename)
        file_extension = f.filename.rsplit('.', 1)[1].lower()

        # check before saving to avoid spamming the folder with invalid files
        validated_extension = self.validate_file_extension(file_extension)
        if validated_extension == 'invalid':
            return jsonify({'error': 'Sorry, this extension is not supported.'}), 406
        
        print("upload folder is ", filepath)

        f.save(filepath)
    
        # initializing classes
        brand = BrandService()
        license_plate = LicensePlateService()

        # section for handling image based recognition
        if validated_extension == 'image':
            encoded_string,vehicle_data,data_brands = self.predict_image(filepath,model,brand,license_plate)
            # remove the image file from the server
            os.remove(filepath)
            if not encoded_string and not vehicle_data and not data_brands:
                return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
            else:
                print(vehicle_data)
                return {'image': encoded_string.decode("utf-8"), 'data_api': jsonify([vehicle.__dict__ for vehicle in vehicle_data]), 'type': 'image', 'data_brands': jsonstring(data_brands)}, 200
                try:
                    return {'image': encoded_string.decode("utf-8"), 'data_api': jsonify(vehicle_data), 'type': 'image', 'data_brands': jsonstring(data_brands)}, 200
                except:
                    return {'error': 'could not decode the image to the response'}, 406
            
        # section for handling video based recognition
        elif validated_extension == 'video':
            decoded_string = self.predict_video(filepath,model,brand,license_plate)
            # remove the file from the server
            os.remove(filepath)
            if not decoded_string:
                return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
            else:
                try:
                    return {'video': jsonstring(decoded_string), 'type': 'video', 'data': ''}, 200
                except:
                    return {'error': 'could not decode the video to the response'}, 406

    # method for handling logic of image prediction
    def predict_image(self, filepath,model,brand,license_plate):
        data_brands = []
        vehicle_data = []
        # read the image
        image = cv2.imread(filepath)

        # check which model is selected
        if model=='BRAND':
            # process brand model 
            image_drawn, data_brands = brand.brand_predict(image, self.yolov5_model)           
        elif model == 'LP':
            # process license plate model
            image_drawn, vehicle_data = license_plate.license_predict(image, self.yolov8_model, self.reader)
        elif model == 'BOTH':
            # process both model
            image_drawn, data_brands = brand.brand_predict(image, self.yolov5_model) 
            image_drawn, vehicle_data = license_plate.license_predict(image_drawn, self.yolov8_model, self.reader)
        else:
            return None,None,None
            
        # Saving the inference image temporarily
        output_image_path = 'output/image.jpg'
        cv2.imwrite(output_image_path, image_drawn)
        with open("output/image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        
        return (encoded_string,vehicle_data,data_brands) 


    # method for handling logic of video prediction
    def predict_video(self,filepath,model,brand,license_plate):
        # check which model is selected
        if model=='BRAND':
            # process brand model           
            img_arr = brand.process_brand_video(filepath, self.yolov5_model)
        elif model == 'LP':
            # process license plate model
            img_arr = license_plate.process_lp_video(filepath, self.yolov8_model,self.reader)
        elif model == 'BOTH':
            # process brand model
            img_arr = brand.process_brand_video(filepath, self.yolov5_model)
                
        else:
            return None

        # create the video from the frames    
        self.construct_video_from_frames(img_arr)
        if model == 'BOTH':
            # process license plate model
            img_arr_lp = license_plate.process_lp_video("output/preprocess_video.mp4", self.yolov8_model,self.reader)
            # create the video from the frames  
            self.construct_video_from_frames(img_arr_lp)
        # video_url = self.upload_video_to_azure("output/preprocess_video.mp4", "preprocess_video.mp4")
        with open("output/preprocess_video.mp4", "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        # decode video
        decoded_string = encoded_string.decode("utf-8")
        
        return decoded_string
        
    # method to check for whether a valid extension has been provided
    def validate_file_extension(self, file_extension):
        image_extensions = ['jpeg', 'jpg', 'png']
        video_extensions = ['mp4', 'mov', 'avi']

        if file_extension in image_extensions:
            return "image"
        elif file_extension in video_extensions:
            return "video"
        else:
            return "invalid"

    # method for building up a video from given frames
    def construct_video_from_frames(self, img_array):
        height, width, _ = img_array[0].shape
        size = (width, height)

        output_path = 'output/preprocess_video.mp4'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

        for i in range(len(img_array)):
            out.write(img_array[i])

