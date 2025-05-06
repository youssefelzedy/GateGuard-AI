import numpy as np

from utils.detectCharacters import getname

# Mapping dictionaries for character conversion
dict_char_to_int = {'a': '1',
                    'g': '2',
                    's': '3',
                    'd': '5',
                    'm': '6',
                    'y': '4',
                    'o': '4'}

dict_int_to_char = {'1': 'a',
                    '2': 'jeem',
                    '3': 's',
                    '4': 'o',
                    '5': '00',
                    '6': 'meem'}

code_en = {'s': 0, 'w': 1, 'tt': 2, 'f': 3, 'a': 4, '3': 5, '9': 6, 'kk': 7, '1': 8, '4': 9, 'ss': 10, 'b': 11, '5': 12,
        'y': 13, 'g': 14, '7': 15, '8': 16, '00': 17, 'd': 18, '2': 19, 'm': 20, 'r': 21, 'l': 22, 'n': 23, '6': 24, 'o': 25}

code_ar = {'س':0,'و':1,'ط':2,'ف':3,'ا':4,'٣':5,'٩':6,'ق':7,'١':8,'٤':9,'ص':10,'ب':11,'٥':12,
           'ي':13,'ج':14,'٧':15,'٨':16,'ه':17,'د':18,'٢':19,'م':20,'ر':21,'ل':22,'ن':23,'٦':24,'ع':25 }


def format_license(num, char):
    """
    Format the license plate text.

    Args:
        num (list): List of detected numbers.
        char (list): List of detected letters.
        char_res (any): Additional character recognition result.

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

def separate_numbers_letters(predictions, width):
    """
    Separate the numbers and letters in the license plate text.

    Args:
        predictions (str): License plate text.
        width (int): Width of the image.


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

def remove_duplicate_boxes(character_bboxes, threshold=5):
    """
    Removes duplicate bounding boxes based on coordinate similarity.

    Args:
        character_bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        threshold (int): Maximum difference between coordinates to consider boxes as duplicates.

    Returns:
        list: Filtered list of bounding boxes with duplicates removed.
    """
    filtered_bboxes = []
    for i, box1 in enumerate(character_bboxes):
        x1_1, y1_1, x2_1, y2_1 = box1
        is_duplicate = False
        for j, box2 in enumerate(filtered_bboxes):
            x1_2, y1_2, x2_2, y2_2 = box2
            # Check if the difference between coordinates is within the threshold
            if (
                abs(x1_1 - x1_2) <= threshold and
                abs(y1_1 - y1_2) <= threshold and
                abs(x2_1 - x2_2) <= threshold and
                abs(y2_1 - y2_2) <= threshold
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_bboxes.append(box1)
    return filtered_bboxes

def get_non_duplicate_predictions(character_bboxes, all_predictions):
    """
    Filters out duplicate bounding boxes and returns sorted predictions.

    Args:
        character_bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        all_predictions (list): List of predictions corresponding to the bounding boxes.
        threshold (int): Maximum difference between coordinates to consider boxes as duplicates.

    Returns:
        list: Sorted list of non-duplicate predictions.
    """

    # Filter predictions corresponding to non-duplicate bounding boxes
    non_duplicate_predictions = []
    for bbox in character_bboxes:
        for prediction in all_predictions:
            if bbox[:4] == prediction[:4]:  # Match bounding box coordinates
                non_duplicate_predictions.append(prediction)
                break

    # Sort predictions by the x-coordinate of the bounding box
    sorted_predictions = sorted(non_duplicate_predictions, key=lambda x: x[0])  # x1 is at index 0
    return sorted_predictions