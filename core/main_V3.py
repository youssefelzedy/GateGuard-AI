from ultralytics import YOLO
from tensorflow import keras

import cv2
import numpy as np
import pandas as pd
from sort.sort import *
import torch

# from utils.util_V2 import get_car, detect_text_yolo, separate_numbers_letters, format_license
from utils.detectCharacters import predict_characters
from utils.segmentCharacters import segment_characters


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./models/yolov8n.pt')
ocr_yolo_model = YOLO('models/yolo11m_car_plate_ocr.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
license_plate_detector2 = YOLO('./models/yolo11m_car_plate_trained.pt')
detect_characters_model = YOLO('./models/characters_detection_new.pt')
new_OCR_model = keras.models.load_model('./models/arabic-OCR-new.h5')

# # load video
# cap = cv2.VideoCapture('./Input-videos/test3.mp4')

vehicles = [2, 3, 5, 7]

def final_model(frame):
    """
    This function processes a video frame to detect vehicles and license plates,
    and then recognizes the characters on the license plates.
    
    Args:
        frame (numpy.ndarray): The video frame to be processed.
        
    Returns:
    tuple: A tuple containing the recognized characters and the license plate text.
    """

    final_reault1 = []

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect the first car in the frame
    first_car = frame[int(
                detections_[0][1]):int(detections_[0][3]), int(detections_[0][0]): int(detections_[0][2]), :]
    output_path = os.path.join("cloped", f"car.png")
    cv2.imwrite(output_path, first_car)



    # detect license plates
    license_plates = license_plate_detector2(frame)[0]

    # Not detect any plate
    if not license_plates.boxes.data.tolist():
        return "Not Detection"

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        char_res = None
        license_plate_text = None

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(
            license_plate, track_ids)

        if car_id != -1:

            # crop license plate
            license_plate_crop = frame[int(
                y1):int(y2), int(x1): int(x2), :]
            license_plate_crop = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(license_plate_crop)
            s = cv2.add(s, 60)
            hsv_enhanced = cv2.merge([h, s, v])
            image_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

            recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions = detect_text_yolo(
                detect_characters_model, image_enhanced
            )
            
            # get character bboxes
            character_bboxes = sorted(character_bboxes, key=lambda x: x[0])
            char_res = segment_characters(image_enhanced, character_bboxes, new_OCR_model)

            output_path = os.path.join("cloped", f"plate.png")
            cv2.imwrite(output_path, image_enhanced)
            # print the size of cropped license plate image = license_plate_crop
            # print("license_plate_crop ==========================")
            # print("Shape:", license_plate_crop.shape)
            # print("===========================================")
            # print("Character BBoxes =====================")
            # print(character_bboxes)
            # print("=====================================")
            # print("Image Text ==========================")
            # print(detected_numbers, detected_letters)
            # print("=====================================")
            # print("All Predictions ======================")
            # print(all_predictions)
            # print("======================================")

            sorted_data = sorted(all_predictions, key=lambda x: x[0])

            # split numbers and letters
            print(x2 - x1)
            num, char = separate_numbers_letters(sorted_data, x2 - x1)

            # format license plate
            license_plate_text = format_license(num, char)
            print("License Plate =======================")
            print(license_plate_text)
            print("=====================================")
        final_reault1.append(char_res)
        final_reault1.append(license_plate_text)


    return final_reault1
