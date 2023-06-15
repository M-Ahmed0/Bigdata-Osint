import cv2

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
            tuple: A tuple containing the processed image with bounding boxes and the predicted brand names.
        """
        results = yolov5_model(image)
        pred_boxes = results.xyxy[0].detach().numpy()

        class_name = ""
        class_names = []
        # For class recognition
        class_labels = results.names

        # Looping through every box detection, identifying coordicates, drawing a box and displaying class name (and returning back the image with class name)
        image_drawn = image.copy()

        for box in pred_boxes:
            xmin, ymin, xmax, ymax, conf, cls = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Class recognition
            class_name = class_labels[int(cls)]

            class_names.append(class_name)

            cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Draw green rectangle
            cv2.putText(image_drawn, class_name, (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2) # Add green text
        

        return image_drawn, class_names


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

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = []

        frame_count = 0

        while ret := video.grab():
            # Retrieve the grabbed frame
            ret, frame = video.retrieve()
            if not ret:
                break

            # Perform object detection and retrieve processed frame with bounding boxes and class names
            processed_frame, class_names = self.brand_predict(frame, yolov5_model)
           
            frame_count += 1

            # Prints to show progress in the console
            if len(class_names) == 0:
                print(f"Processing frame ({frame_count}/{total_frames}): no detections found")
            else:
                print(f"Processing frame ({frame_count}/{total_frames}): {' '.join(class_names)}")
            
            processed_frames.append(processed_frame)

        # Release the video capture object
        video.release()
        print('Brand video has been processed!')

        return processed_frames
