import torch
from ultralytics import YOLO
import cv2
from gradio_client import Client, handle_file
from pathlib import Path
import tempfile
import json

def select_pytorch_device():
    """
    Selects the most appropriate device for PyTorch operations.
    Prioritizes CUDA, then MPS, and finally falls back to CPU.

    Returns:
        str: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
      
class LicensePlate(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize the models and processors
        self.device = select_pytorch_device()
        # Load the YOLO model 
        self.license_model = YOLO(r"C:/Workspace/chungvodim/model/yolov8/train8/weights/best.pt").to(self.device)
        self.client = Client("gokaygokay/Florence-2")

    def run_ocr(self, image):
        # Create tempfile as florence ocr requires image filepath or url to run 
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, image)
        temp_file.close()
        # Run ocr on cropped licence plate
        result = self.client.predict(
                        image=handle_file(temp_file_path),
                        task_prompt="OCR",
                        text_input=None,
                        model_id="microsoft/Florence-2-large",
                        api_name="/process_image"
                        )
        Path(temp_file_path).unlink() # Delete tempfile
        # Format OCR result
        ocr_result_dict = json.loads(result[0].replace("'", '"')) 
        ocr_text = ocr_result_dict['<OCR>']
        return ocr_text
    
    def detect_license(self, image):
        results = self.license_model.predict( 
                        task="detect",
                        source=image,
                        save=False,  # Do not save predictions
                        conf=0.5,
                        device=self.device
                        )
        annotated_image = results[0].plot() #Gives annotated frame with bbox around detections, class and conf values

        for r in results:
            # Check if there are any detections
            if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
                for coordinates in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, coordinates)
                    cropped_image = image[y1:y2, x1:x2]
                    #Run Florence on image and get OCR text
                    ocr_text = self.run_ocr(cropped_image)
                    # Overlay OCR text on the annotated frame with white bg
                    font_scale = 1  
                    thickness = 2
                    text_size, _ = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_w, text_h = text_size
                    cv2.rectangle(annotated_image, (x2, y2 - text_h - 10), (x2 + text_w, y2 + 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(annotated_image, ocr_text, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)       
        
        return annotated_image
    
        
