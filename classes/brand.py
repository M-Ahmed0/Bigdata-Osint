import cv2
from VehicleDTO import VehicleDTO

class Brand:
    def brand_predict(self, image, yolov5_model):
        results = yolov5_model(image)
        pred_boxes = results.xyxy[0].detach().numpy()

        class_name = ""
        
        # For class recognition
        class_names = results.names

        image_drawn = image.copy()
        for box in pred_boxes:
            xmin, ymin, xmax, ymax, conf, cls = box
            print("xmin", xmin, "ymin", ymin, "xmax", xmax, "ymax", ymax) 
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Class recognition
            class_name = class_names[int(cls)]
            print("Brand detected:", class_name)

            cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Draw green rectangle
            cv2.putText(image_drawn, class_name, (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) # Add green text
        
        vehicle_dto = VehicleDTO.brand_dto(class_name)

        return image_drawn, vehicle_dto


    def process_brand_video(self, video_path, yolov5_model):
        video = cv2.VideoCapture(video_path)

        processed_frames = []

        while ret := video.grab():
            # Retrieve the grabbed frame
            ret, frame = video.retrieve()
            if not ret:
                break

            # Perform object detection and retrieve processed frame with bounding boxes and class names
            processed_frame, _ = self.brand_predict(frame, yolov5_model)
            
            processed_frames.append(processed_frame)

        # Release the video capture object
        video.release()
        print('Video processing completed')

        return processed_frames
