from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def detect_text_yolo(model, cropped_image):
    """
    Detect text in the cropped image using YOLOv8.
    Args:
        model (YOLO): YOLOv8 model.
        cropped_image (numpy.ndarray): Cropped image of the license plate.
    Returns:
        recognized_image (numpy.ndarray): Image with detected text.
        detected_numbers (list): List of detected numbers.
        detected_letters (list): List of detected letters.
        character_bboxes (list): List of bounding boxes for each character.
        all_predictions (list): List of all predictions with bounding boxes and text.
    """

    # Run prediction
    results = model.predict(cropped_image, conf=0.25)

    # print("RESULTS ==========================")
    # for r in results:
    #     boxes = r.boxes
    #     names = r.names  # dict: {id: class_name}
    #     for box in boxes:
    #         cls_id = int(box.cls[0])  # رقم الكلاس
    #         conf = float(box.conf[0])  # النسبة
    #         xyxy = box.xyxy[0].tolist()  # إحداثيات البوكس
    #         print(f"Detected: {names[cls_id]} - Confidence: {conf:.2f} - Box: {xyxy}")
    # print("==================================")

    detected_numbers = []
    detected_letters = []
    character_bboxes = []
    all_predictions = []

    # Get class names
    possible_classes = results[0].names

    # Get detected boxes
    boxes = results[0].boxes

    # Check if any boxes were detected
    if boxes is None or len(boxes) == 0:
        return None, [], [], [], []  # Return empty values if no detections

    # Convert image to numpy for visualization
    recognized_image = np.array(results[0].plot())
    recognized_image = cv2.cvtColor(recognized_image, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])  # Bounding box coordinates
        conf = boxes.conf[i].item()  # Confidence score
        class_id = int(boxes.cls[i].item())  # Predicted class index
        recognized_text = possible_classes[class_id]  # Class name

        # Store bounding box info
        character_bboxes.append([x1, y1, x2, y2])

        # **Extract Top-3 Predictions for this Bounding Box**
        char_options = []
        for j in range(len(boxes.cls)):
            if j == i:  # Only consider the current box's predictions
                cls_id = int(boxes.cls[j].item())
                cls_conf = boxes.conf[j].item()
                char_options.append((possible_classes[cls_id], cls_conf))

        # Sort by confidence and keep the top 3
        char_options_sorted = sorted(
            char_options, key=lambda x: x[1], reverse=True)[:3]

        formatted_options = " / ".join(
            [f"{char} ({score:.2f})" for char, score in char_options_sorted])

        # Separate numbers and letters
        if any(char.isdigit() for char, _ in char_options_sorted):
            detected_numbers.append(formatted_options)
        else:
            detected_letters.append(formatted_options)

        # Store all possibilities
        all_predictions.append([x1, y1, x2, y2, formatted_options])

    return recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
