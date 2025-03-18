import torch
import string
import easyocr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# Initialize the OCR reader
reader = easyocr.Reader(['ar'], gpu=False)

yolo_model = 'models/yolo11m_car_plate_trained.pt'

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

import cv2
import numpy as np
from ultralytics import YOLO

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
                # print("=====================================")
                # print(f"Box {i}: {boxes.cls[j].item()}")
                # print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
                cls_id = int(boxes.cls[j].item())
                cls_conf = boxes.conf[j].item()
                char_options.append((possible_classes[cls_id], cls_conf))

        # print("ooooooooooooooooooooooooooooooooooooooooo")
        # Sort by confidence and keep the top 3
        char_options_sorted = sorted(char_options, key=lambda x: x[1], reverse=True)[:3]  
        
        formatted_options = " / ".join([f"{char} ({score:.2f})" for char, score in char_options_sorted])
        
        # Separate numbers and letters
        if any(char.isdigit() for char, _ in char_options_sorted):
            detected_numbers.append(formatted_options)
        else:
            detected_letters.append(formatted_options)

        # Store all possibilities
        all_predictions.append([x1, y1, x2, y2, formatted_options])

        # **Draw bounding box and text on the image**
        cv2.rectangle(recognized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(recognized_image, formatted_options, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display text

        # Print the formatted options for debugging
        # print(f"Box {i}: ({x1}, {y1}, {x2}, {y2}) → {formatted_options}")

    return recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions



# Function to crop the lower part of the detected plate with increased width and adjustment above midpoint
def crop_LowerPart_Plate(image, width_margin=20, y_offset=5):
    model = YOLO(yolo_model)

    # Perform prediction on the image with a confidence threshold of 0.25
    results = model.predict(source=image, conf=0.25)

    # # Open the image
    # image = Image.open(img)

    # Iterate over all the results
    for result in results:
        # Ensure boxes are detected
        if result.boxes is not None and len(result.boxes) > 0:
            max_width = -1
            selected_box = None

            # Iterate through each detected bounding box
            for box in result.boxes:
                # Get the bounding box coordinates: [x_min, y_min, x_max, y_max]
                res = box.xyxy[0]
                # Calculate width: (x_max - x_min)
                width = res[2].item() - res[0].item()

                # Update if the current box is the widest one
                if width > max_width:
                    max_width = width
                    selected_box = res  # Store the coordinates of the widest box

            # Once the widest box is found, proceed with cropping
            if selected_box is not None:
                # Adjust the bounding box coordinates
                # Decrease x_min for more width
                x_min = int(selected_box[0].item()) - width_margin
                y_min = int(selected_box[1].item())  # Start above the midpoint
                # Increase x_max for more width
                x_max = int(selected_box[2].item()) + width_margin
                y_max = int(selected_box[3].item())  # Keep y_max as is

                # Ensure the coordinates are within image bounds (optional check)
                img_width, img_height = image.size
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)

                # Debug: Print the bounding box coordinates
                print(
                    f"Cropping coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                # Crop the image using the adjusted bounding box
                cropped_image = image.crop((x_min, y_min, x_max, y_max))

                # cropped_image_path = 'cropped_plate_image.jpg'  # Specify the path to save the cropped image
                # processed_image.save(cropped_image_path)

                # Resize the cropped image to a standard size (130x130)
                # resized_cropped_image = cropped_image.resize((100, 130))

                # Return the final image (cropped and resized)
                return cropped_image

        else:
            print("No bounding boxes detected.")
    return None



# def display_predictions(image, character_bboxes, all_predictions):
#     for i, (box, predictions) in enumerate(zip(character_bboxes, all_predictions)):
#         # Debugging output
#         print(f"Box {i}: {box}, Predictions: {predictions}")

#         x1, y1, x2, y2 = box
#         cv2.rectangle(image, (x1, y1), (x2, y2),
#                       (0, 255, 0), 2)  # Draw bounding box

#         # Ensure `predictions` is a list of tuples
#         if isinstance(predictions, tuple) and len(predictions) == 2 and isinstance(predictions[0], str):
#             # Convert single tuple to a list of tuples
#             predictions = [predictions]

#         for j, (char, conf) in enumerate(predictions):
#             text = f"{char} ({conf:.2f})"
#             text_position = (x1, y1 - 10 - (j * 15))  # Offset each line
#             cv2.putText(image, text, text_position,
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     cv2.imshow("Character Predictions", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import cv2
from paddleocr import PaddleOCR, draw_ocr

def detect_plate_text(license_plate_crop):
    # Initialize PaddleOCR model (Arabic support)
    ocr = PaddleOCR(use_angle_cls=True, lang="ar")  

    # Run OCR on the cropped license plate
    results = ocr.ocr(license_plate_crop, cls=True)

    detected_texts = []
    detected_bboxes = []

    print("RESULTS ==========================")
    print(results)
    print("==================================")
    if results[0] is None:
        return None, None
    
    for result in results:
        for line in result:
            box, (text, confidence) = line
            
            # Save results
            detected_bboxes.append(box)
            detected_texts.append(f"{text} ({confidence:.2f})")

    # Draw bounding boxes on the cropped image
    # boxed_image = draw_ocr(license_plate_crop, detected_bboxes, detected_texts, font_path="arabic.ttf")

    return detected_texts, detected_bboxes

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.replace('', ' ')
        print("TEXT ========================")
        print(text, score)
        print("========================")

        # if license_complies_format(text):
        return text, score

    return None, None


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
