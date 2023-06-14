import re
import cv2
from dtos.VehicleDTO import VehicleDTO
import requests
import numpy as np
import configparser


class LicensePlateService:
    """
    A service class for license plate detection and recognition.

    Attributes:
        app_token (str): App token for RDW API.

    Methods:
        license_predict(image, yolov8_model, reader):
            Performs license plate prediction on a single image using the YOLOv8 model.

        process_lp_video(video, yolov8_model, reader):
            Processes license plate frames in a video using the YOLOv8 model.

        get_vehicle_data(license_plate):
            Retrieves vehicle data from the RDW API based on the license plate.

        process_results(results, image, reader):
            Processes the detection results by enhancing pixels and performing OCR to read license plates.

        process_lp_video_frames(results, reader):
            Processes license plate frames in a video by enhancing pixels and performing OCR.

        clean_license_plate(license_plate):
            Cleans the license plate string by removing special characters and converting to uppercase.

        read_text_ocr(img, reader):
            Performs OCR on an image and returns the extracted text.
    """


    # Load the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # get this API token
    app_token = config.get('API', 'app_token')

    # get the BASE_URL 
    base_url = config.get('BASE_URL', 'base_url')
    
    def license_predict(self,image,yolov8_model,reader):
        """
        Performs license plate prediction on a single image using the YOLOv8 model.

        Args:
            image (numpy.ndarray): Input image for license plate detection.
            yolov8_model: Pre-trained YOLOv8 model for object detection.
            reader: OCR reader for text extraction.

        Returns:
            tuple: A tuple containing the original image, vehicle data, and detected license plate.
        """
        # Perform license plate prediction using the YOLOv8 model
        detections = yolov8_model.predict(image)  
        license_plate,image_drawn,vehicle_data = self.process_results(detections,image,reader)

        # Save the drawn image as PNG ----- testing purposes
        output_path = 'test.png'
        cv2.imwrite(output_path, cv2.cvtColor(image_drawn, cv2.COLOR_RGB2BGR))
        return (image,vehicle_data)
    

    def process_lp_video(self,video,yolov8_model,reader):
        """
        Processes license plate frames in a video using the YOLOv8 model.

        Args:
            video: Input video for license plate detection.
            yolov8_model: Pre-trained YOLOv8 model for object detection.
            reader: OCR reader for text extraction.

        Returns:
            list: A list of processed frames with license plates highlighted.
        """
        # Process license plate frames in a video using the YOLOv8 model
        detections = yolov8_model.predict(video)
        processed_frames = self.process_lp_video_frames(detections,reader)
        return processed_frames
    
    def get_vehicle_data(self,license_plate): 
        """
        Retrieves vehicle data from the RDW API based on the license plate.

        Args:
            license_plate (str): License plate number.

        Returns:
            dict: Vehicle data as a dictionary.
        """
        params = { 'kenteken': license_plate, '$$app_token': self.app_token } 
        try: 
            response = requests.get(self.base_url, params=params) 
            response.raise_for_status() # Raise an exception for 4xx or 5xx errors 
            data = response.json() 
            if len(data) ==0:
                return data
            vehicle_dto = VehicleDTO.from_json(data)
            return vehicle_dto 
    
        except requests.exceptions.RequestException as e: 
            print(f"An error occurred: {e}")
    
    
    def process_results(self,results, image,reader):
        """
        Processes the detection results by enhancing pixels and performing OCR to read license plates.

        Args:
            results: Object detection results.
            image (numpy.ndarray): Original image with detected objects.
            reader: OCR reader for text extraction.

        Returns:
            tuple: A tuple containing the detected license plate, the annotated image, and vehicle data.
        """
        # process all the results by enhancing the pixels and performing OCR to read the license plates.
        try:
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
                            
                    # read text using ocr
                    text = self.read_text_ocr(thresh, reader)

                    vehicle_data = self.get_vehicle_data(text)
                    if vehicle_data=="":
                        # Create a kernel for dilation and erosion
                        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
                        # Apply dilation
                        dilated_image = cv2.dilate(img, kernel, iterations=1)
                        # Apply erosion
                        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
                        #ocr
                        text = self.read_text_ocr(eroded_image, reader)
                        vehicle_data = self.get_vehicle_data(text)

                    print(text)

                    # draw a rectangle around the box   
                    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) 
                
                    # put the text above the box in the image
                    cv2.putText(image, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) 

                    return (text,image,vehicle_data)
        except:
            pass

        # Return default values when no results or error occurred
        return ("", image, "")
    
    
    def process_lp_video_frames(self,results,reader):
        """
        Processes license plate frames in a video by enhancing pixels and performing OCR.

        Args:
            results: Object detection results.
            reader: OCR reader for text extraction.

        Returns:
            list: A list of processed frames with license plates highlighted.
        """
        processed_frames = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
            
                x,y,w,h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                img = r.orig_img[y:h , x:w]

                # apply grayscale             
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
            
                # Apply threshold
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                           
                # read text using ocr
                text = self.read_text_ocr(thresh, reader)
                print(text)
                
                # draw a rectangle around the box   
                cv2.rectangle(r.orig_img, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle

                # put the text above the box in the image
                cv2.putText(r.orig_img, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text
            
            processed_frames.append(r.orig_img)

    
        return processed_frames

    # ---- clean the text by removing unwanted characters
    def clean_license_plate(self,license_plate): 
        """
        Cleans the license plate string by removing special characters and converting to uppercase.

        Args:
            license_plate (str): License plate number.

        Returns:
            str: Cleaned license plate number.
        """
        cleaned_plate = re.sub(r"[^a-zA-Z0-9]", "", license_plate)
        return cleaned_plate.upper()

    # ---- read the text using ocr reader
    def read_text_ocr(self,img,reader):
        """
        Performs OCR on an image and returns the extracted text.

        Args:
            img (numpy.ndarray): Input image for OCR.
            reader: OCR reader for text extraction.

        Returns:
            str: Extracted text.
        """
        #ocr
        result = reader.readtext(img)                   
        text = ""
        
        # process ocr results to form the text
        for res in result:
            if len(result) == 1:
                text = res[1]
        
            if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
                text = res[1]
        text = self.clean_license_plate(text)
        return text
#---------------------------------------------------------------------------------------------------------------------------------------#
