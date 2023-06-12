import cv2
from dtos.VehicleDTO import VehicleDTO

class BrandService:
    """
    Service class for brand recognition using object detection models.

    Methods:
        brand_predict(image, yolov5_model): Performs brand prediction on a single image.
        process_brand_video(video_path, yolov5_model): Processes a brand recognition on a video.

    """
    def brand_predict(self, image, yolov5_model):
        """
        Performs brand prediction on a single image using a YOLOv5 model.

        Args:
            image (numpy.ndarray): Input image for brand prediction.
            yolov5_model: YOLOv5 model for object detection.

        Returns:
            tuple: A tuple containing the processed image with bounding boxes and the predicted brand name.
        """
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
        

        return image_drawn, class_name


    def process_brand_video(self, video_path, yolov5_model):
        """
        Processes brand recognition on a video using a YOLOv5 model.

        Args:
            video_path (str): Path to the input video.
            yolov5_model: YOLOv5 model for object detection.

        Returns:
            list: A list of processed frames with bounding boxes and class names.
        """
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
