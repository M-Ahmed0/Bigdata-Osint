from json import dumps as jsonstring
import cv2
from flask import jsonify
from services.brand_service import BrandService
from services.license_plate_service import LicensePlateService
import os
import base64
from azure.storage.blob import BlobServiceClient, ContentSettings ,generate_container_sas, ContainerSasPermissions, BlobSasPermissions, AccountSasPermissions
from datetime import datetime, timedelta
import pytz

class PredictController:
    # Azure Blob Storage connection string
    connection_string = "AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;DefaultEndpointsProtocol=http;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"

    # Container name in Azure Blob Storage
    container_name = "prediction"
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
            encoded_string,vehicle_data,data_brand = self.predict_image(filepath,model,brand,license_plate)
            # remove the image file from the server
            os.remove(filepath)
            if not encoded_string and not vehicle_data and not data_brand:
                return jsonify({'error': 'Sorry, the selected model does not exist.'}), 406
            else:
                try:
                    return {'image': encoded_string.decode("utf-8"), 'data_api': jsonstring('' if not vehicle_data else vehicle_data.__dict__ ), 'type': 'image', 'data_brand': jsonstring(data_brand)}, 200
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
        data_brand = ''
        vehicle_data = []
        # read the image
        image = cv2.imread(filepath)

        # check which model is selected
        if model=='BRAND':
            # process brand model 
            image_drawn, data_brand = brand.brand_predict(image, self.yolov5_model)           
        elif model == 'LP':
            # process license plate model
            image_drawn, vehicle_data = license_plate.license_predict(image, self.yolov8_model, self.reader)
        elif model == 'BOTH':
            # process both model
            image_drawn, data_brand = brand.brand_predict(image, self.yolov5_model) 
            image_drawn, vehicle_data = license_plate.license_predict(image_drawn, self.yolov8_model, self.reader)
        else:
            return None,None,None
            
        # Saving the inference image temporarily
        output_image_path = 'output/image.jpg'
        cv2.imwrite(output_image_path, image_drawn)
        with open("output/image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        
        return (encoded_string,vehicle_data,data_brand) 


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
        video_url = self.upload_video_to_azure("output/preprocess_video.mp4", "preprocess_video.mp4")
        # with open("output/preprocess_video.mp4", "rb") as video_file:
        #     encoded_string = base64.b64encode(video_file.read())
        # # decode video
        # decoded_string = encoded_string.decode("utf-8")
        print(video_url)
        print(video_url)
        print(video_url)
        print(video_url)
        print(video_url)
        return video_url
        
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


    def upload_video_to_azure(self,file_path, file_name):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.container_name)
        #permissions = ContainerSasPermissions(read=True)
        permissions = AccountSasPermissions(read=True)
        # Define the desired time zone
        # timezone = pytz.timezone('Europe/Paris')

        # # Get the current UTC time
        # current_time = datetime.utcnow()

        # # Convert the current UTC time to the desired time zone
        # current_time_europe = current_time.astimezone(timezone)

        # # Set the expiry time as 10 hours from the current time in Europe time
        # expiry = current_time_europe + timedelta(hours=10)
        expiry = datetime.utcnow() + timedelta(hours=10)
        sas_token = generate_container_sas(
            account_name=blob_service_client.account_name,
            container_name=self.container_name,
            account_key=blob_service_client.credential.account_key,
            permission=permissions,
            expiry=expiry
            )
        

        with open(file_path, "rb") as data:
            container_client.upload_blob(name=file_name, data=data,overwrite=True, content_settings=ContentSettings(content_type='video/mp4'))
        # Return the video URL
        # Include the SAS token in the video URL
        video_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{file_name}?{sas_token}"
        return video_url