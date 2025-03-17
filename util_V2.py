import torch
import string
import easyocr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Mapping dictionaries for character conversion
dict_char_to_int = {'alif': '1',
                    'jeem': '2',
                    'seen': '3',
                    'ain': '4',
                    'haa': '5',
                    'meem': '6'}

dict_int_to_char = {'1': 'alif',
                    '2': 'jeem',
                    '3': 'seen',
                    '4': 'ain',
                    '5': 'haa',
                    '6': 'meem'}


# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.

#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
#                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#                                                 'license_number_score'))

#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and
#                 'license_plate' in results[frame_nmr][car_id].keys() and
#                 'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                             car_id,
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                             results[frame_nmr][car_id]['license_plate']['text'],
#                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
#                             )
#         f.close()


# def license_complies_format(text):
#     """
#     Check if the license plate text complies with the required format.

#     Args:
#         text (str): License plate text.

#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     if len(text) != 7:
#         return False

#     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
#     (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
#     (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and
#     (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and
#     (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
#     (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
#     (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
#         return True
#     else:
#         return False

def separate_numbers_letters(predictions, width):
    """
    Separate the numbers and letters in the license plate text.

    Args:
        predictions (str): License plate text.
        image (PIL.Image): License plate image.


    Returns:
        tuple: Tuple containing the separated numbers and letters.
    """

    # initialize lists
    num = []
    char = []

    # Separate numbers and letters
    for prediction in predictions:
        x1, y1, x2, y2, text = prediction
        if x1 < width/2:
            num.append(prediction)
        else:
            char.append(prediction)

    return num, char


def format_license(num, char):
    """
    Format the license plate text.

    Args:
        num (list): List of detected numbers.
        char (list): List of detected letters.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''

    # Map the detected characters to the correct format
    for i in range(len(num)):
        x1, y1, x2, y2, text = num[i]
        
        cleaned_text = text.split("(")[0].strip()
        if cleaned_text in dict_char_to_int.keys():
            license_plate_ += dict_char_to_int[cleaned_text]
        else:
            license_plate_ += cleaned_text

        license_plate_ += '-'

    for i in range(len(char)):
        x1, y1, x2, y2, text = char[i]
        
        cleaned_text = text.split("(")[0].strip()
        if cleaned_text in dict_int_to_char.keys():
            license_plate_ += dict_int_to_char[cleaned_text]
        else:
            license_plate_ += cleaned_text

        if i != len(char) - 1:
            license_plate_ += '-'

    return license_plate_


def detect_text_yolo(ocr_yolo_model, cropped_image):
    # Load the YOLO model
    model = YOLO(ocr_yolo_model)

    # Run prediction
    results = model.predict(source=cropped_image, conf=0.25)

    # print("RESULTS ==========================")
    # print(results[0].boxes)
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

        # print("ooooooooooooooooooooooooooooooooooooooooo")
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

        # **Draw bounding box and text on the image**
        # cv2.rectangle(recognized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        # cv2.putText(recognized_image, formatted_options, (x1, y1 - 10),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display text

        # Print the formatted options for debugging
        # print(f"Box {i}: ({x1}, {y1}, {x2}, {y2}) → {formatted_options}")

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
