from ultralytics import YOLO
from tensorflow import keras

import cv2
import numpy as np
import pandas as pd
from sort.sort import *

from utils.detectPlateCharacters import get_car, detect_text_yolo
from utils.detectCharacters import predict_characters
from utils.segmentCharacters import segment_characters
from utils.licenseFormat import remove_duplicate_boxes, get_non_duplicate_predictions, separate_numbers_letters, format_license



results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./models/yolov8n.pt')
license_plate_detector2 = YOLO('./models/yolo11m_car_plate_trained.pt')
detect_characters_model = YOLO('./models/best.pt')
# new_OCR_model = keras.models.load_model('./models/arabic-OCR-new-with-3-dataset-last-v2.h5')

vehicles = [2, 3, 5, 7]

def final_model(frame):
    """
    This function processes a video frame to detect vehicles and license plates,
    and then recognizes the characters on the license plates.
    
    Args:
        frame (numpy.ndarray): The video frame to be processed.
        
    Returns:
    tuple: A tuple containing the recognized characters and the license plate text and palte, car detections.
    """
    try:
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        final_results = []
        car_id = 1
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, car_id])
                car_id += 1

        # track vehicles
        # track_ids = mot_tracker.update(np.asarray(detections_))
        
        # Check if there are any detections
        if len(detections_) == 0:
            return "No Detection"

        # Ensure the first detection has valid bounding box coordinates
        if len(detections_[0]) < 4:
            return "Invalid detection format"

        # Detect the first car in the frame
        try:
            first_car = frame[int(
                detections_[0][1]):int(detections_[0][3]), int(detections_[0][0]): int(detections_[0][2]), :]
        except IndexError:
            return "Invalid bounding box indices"

        # Save the cropped car image
        if first_car is None or first_car.size == 0:
            return "No valid car detected"

        # output_path = os.path.join("cropped", f"car.png")
        # cv2.imwrite(output_path, first_car)

        # detect license plates
        license_plates = license_plate_detector2(frame)[0]

        # Check if any license plates are detected
        if not license_plates.boxes.data.tolist():
            return "No Detection"

        for license_plate in license_plates.boxes.data.tolist():
            char_res = None
            license_plate_text = None
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, detections_
            )

            if car_id != -1:
                # Process the license plate
                try:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                except IndexError:
                    final_results.append(["Invalid bounding plate box indices"])
                    continue
                
                license_plate_crop = cv2.cvtColor(
                    license_plate_crop, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(license_plate_crop)
                s = cv2.add(s, 60)
                hsv_enhanced = cv2.merge([h, s, v])
                image_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

                recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions = detect_text_yolo(
                    detect_characters_model, image_enhanced
                )
                
                if not (len(detected_numbers) >= 1 and len(detected_letters) <= 4):
                    final_results.append(["Invalid license plate format"])
                    continue
                if not (len(detected_letters) >= 1 and len(detected_letters) <= 3):
                    final_results.append(["Invalid character bounding boxes"])
                    continue

                # get character bboxes
                character_bboxes = sorted(character_bboxes, key=lambda x: x[0])
                character_bboxes = remove_duplicate_boxes(character_bboxes, threshold=5)
                # char_res = segment_characters(image_enhanced, character_bboxes, new_OCR_model)

                # output_path = os.path.join("cropped", f"plate.png")
                # cv2.imwrite(output_path, image_enhanced)

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
                # print("Numbers ==========================")
                # print(num)
                # print("=====================================")
                # print("Characters ==========================")
                # print(char)
                # print("=====================================")

                # Ensure all_predictions is not empty
                if not all_predictions:
                    final_results.append(["No predictions found."])
                    continue

                # Get non-duplicate predictions
                sorted_predictions = get_non_duplicate_predictions(character_bboxes, all_predictions)

                if not sorted_predictions:
                    final_results.append(["No sorted predictions found."])
                    continue

                # split numbers and letters
                num, char = separate_numbers_letters(sorted_predictions, x2 - x1)

                # format license plate
                license_plate_text = format_license(num, char)
                final_results.append([license_plate_text, [xcar1, ycar1, xcar2, ycar2, car_id], [x1, y1, x2, y2, score, class_id]])

                print("License Plate =======================")
                print(license_plate_text)
                print("=====================================")
            else:
                print("No car detected for this license plate.")
                final_results.append(["No car detected for this license plate."])
                continue
        return final_results
    except Exception as e:
        print("Error in final_model:", e)
        return "Error in final_model"
