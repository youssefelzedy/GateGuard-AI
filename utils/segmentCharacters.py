import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.detectCharacters import predict_characters


def segment_characters(license_plate_crop, character_bboxes, model):
    """
    Segments characters from the cropped license plate image and saves them to a folder.
    Args:
        license_plate_crop (numpy.ndarray): Cropped license plate image.
        character_bboxes (list): List of bounding boxes for each character.
        model: OCR model for character prediction.
    Returns:
        None
    """

    text = []
    # Create the 'cloped' folder if it doesn't exist
    output_folder = "cloped"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (cx1, cy1, cx2, cy2) in enumerate(character_bboxes):

        # Adjust coordinates relative to the cropped license plate image
        cropped_character = license_plate_crop[int(
            cy1-5):int(cy2+5), int(cx1-1):int(cx2+1)]

        # Convert to grayscale (improves OCR accuracy)
        gray_char = cv2.cvtColor(
            cropped_character, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray_char, (3, 3), 0)

        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2  # Try increasing blockSize to 11 or 15
        )

        # Save the cropped character to the 'cloped' folder
        output_path = os.path.join(output_folder, f"char_{i}.png")
        
        # Ensure unique filenames by appending a number if the file already exists
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_folder, f"char_{i}_{counter}.png")
            counter += 1
        
        cv2.imwrite(output_path, blurred)

        # # Apply Laplacian filter to enhance edges
        # laplacian = cv2.Laplacian(
        #     blurred, cv2.CV_64F)  # Compute Laplacian
        # # Convert back to uint8
        # laplacian = np.uint8(np.absolute(laplacian))

        characters = predict_characters(model, blurred)

        # Append the predicted character to the list
        text.append(characters)
    print(f"Character text: {text}")

    if len(text) == 0:
        return None

    return text
