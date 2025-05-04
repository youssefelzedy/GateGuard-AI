import torch
import string
import easyocr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image

code = {'س': 0, 'و': 1, 'ط': 2, 'ف': 3, 'ا': 4, '٣': 5, '٩': 6, 'ق': 7, '١': 8, '٤': 9, 'ص': 10, 'ب': 11, '٥': 12,
        'ي': 13, 'ج': 14, '٧': 15, '٨': 16, 'ه': 17, 'د': 18, '٢': 19, 'م': 20, 'ر': 21, 'ل': 22, 'ن': 23, '٦': 24, 'ع': 25}


def getname(n):
    """
    Get the name of the character from its code.
    This function takes a character code as input and returns the corresponding character name.
    The mapping is defined in the 'code' dictionary.
    Args:
          n (int): The character code.
    Returns:
          str: The name of the character.
   """
    for k, v in code.items():
        if v == n:
            return k


def predict_characters(model_n, img_1):
    """Predict the character in the image using the trained model.
    This function takes the image and a trained model as input, preprocesses the image,
    and predicts the character using the model. The predicted character is then displayed.
    Args:
        model_n (torch.nn.Module): The trained model for character recognition.
        path (str): The path to the image file.
    Returns:
        list: A list of predicted characters and their probabilities.
    """
    results = []

    img_1 = cv2.resize(img_1, (32, 32))
    img_1 = np.array(img_1)
    img_1_3d = img_1.reshape((1, 32, 32))

    img_1_prob = model_n.predict(np.array(img_1_3d))

    top_4_indices = img_1_prob.argsort(axis=1)[:, -4:][:, ::-1]
    top_4_probabilities = img_1_prob[0][top_4_indices[0]]

    print("Top 4 predictions:", getname(top_4_indices[0][0]), getname(top_4_indices[0][1]), getname(top_4_indices[0][2]))


    for i in range(4):
        results.append([getname(top_4_indices[0][i]), top_4_probabilities[i]])
    
    print("Top 4 predictions with probabilities:", results)
    return results
