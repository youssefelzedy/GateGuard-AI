import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.detectCharacters import predict_character


def segment_characters(license_plate_crop, character_bboxes, model):
    """
    Segments characters from the cropped license plate image.
    Args:
        license_plate_crop (numpy.ndarray): Cropped license plate image.
        character_bboxes (list): List of bounding boxes for each character.
    Returns:
        the characters segmented from the license plate image.
    """

    for i, (cx1, cy1, cx2, cy2) in enumerate(character_bboxes):

        # Adjust coordinates relative to the cropped license plate image
        cropped_character = license_plate_crop[int(
            cy1-5):int(cy2+5), int(cx1-5):int(cx2+5)]

        # Convert to grayscale (improves OCR accuracy)
        gray_char = cv2.cvtColor(
            cropped_character, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_char, (3, 3), 0)
        thresholded = cv2.adaptiveThreshold(
            gray_char,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            61, 2  # Try increasing blockSize to 11 or 15
        )

        # Apply Laplacian filter to enhance edges
        laplacian = cv2.Laplacian(
            blurred, cv2.CV_64F)  # Compute Laplacian
        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))

        characters = predict_character(model, thresholded)

        print(f"Character {i}: {characters}")
