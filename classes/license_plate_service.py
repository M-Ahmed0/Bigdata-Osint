import re
import cv2
from VehicleDTO import VehicleDTO
import requests
import numpy as np



class license_plate_service:

    # Define app token for RDW API
    app_token = 'vFOK1jHTLo7llP150mPktYWgJ'

    def license_predict(self,image,yolov8_model,reader):

        detections = yolov8_model.predict(image)  
        license_plate,image_drawn,vehicle_data = self.process_results(detections,image,reader)

        # Save the drawn image as PNG ----- testing purposes
        output_path = 'test.png'
        cv2.imwrite(output_path, cv2.cvtColor(image_drawn, cv2.COLOR_RGB2BGR))
        return (image,vehicle_data)
    

    def process_lp_video(self,video,yolov8_model,reader):
        detections = yolov8_model.predict(video)
        processed_frames = self.process_lp_video_frames(detections,reader)
        return processed_frames
    
    def get_vehicle_data(self,license_plate): 
        base_url = 'https://opendata.rdw.nl/resource/m9d7-ebf2.json' 
        params = { 'kenteken': license_plate, '$$app_token': self.app_token } 
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
    
    # ---- process all the results by enhancing the pixels and performing OCR to read the license plates.
    def process_results(self,results, image,reader):
    
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
                   
                cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
                cv2.putText(image, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text

                return (text,image,vehicle_data)
    
    # ---- process all the results by enhancing the pixels and performing OCR to read the license plates.
    def process_lp_video_frames(self,results,reader):
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
                           
                
                text = self.read_text_ocr(thresh, reader)
                print(text)
                   
                cv2.rectangle(r.orig_img, (x, y), (w, h), (0, 255, 0), 2) # draw green rectangle
            
                cv2.putText(r.orig_img, text, (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2) # add green text
                processed_frames.append(r.orig_img)

            #return (text,r.orig_img)
    
        return processed_frames


    def clean_license_plate(self,license_plate): 
        cleaned_plate = re.sub(r"[^a-zA-Z0-9]", "", license_plate)
        return cleaned_plate.upper()

    def read_text_ocr(self,img,reader):
        #ocr
        result = reader.readtext(img)                   
        text = ""
        
        for res in result:
            if len(result) == 1:
                text = res[1]
        
            if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
                text = res[1]
        text = self.clean_license_plate(text)
        return text
#---------------------------------------------------------------------------------------------------------------------------------------#
