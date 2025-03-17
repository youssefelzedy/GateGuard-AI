from ultralytics import YOLO
import cv2

import util_V1
from sort.sort import *
from util_V1 import get_car, write_csv, detect_text_yolo
import torch

import numpy as np
import pandas as pd


results = {}

mot_tracker = Sort()


# load models
coco_model = YOLO('./models/yolov8n.pt')
ocr_yolo_model = 'models/yolo11m_car_plate_ocr.pt'
license_plate_detector = YOLO('./models/license_plate_detector.pt')
license_plate_detector2 = YOLO('./models/yolo11m_car_plate_trained.pt')

# load video
cap = cv2.VideoCapture('./Input-videos/test10.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector2(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(
                    y1):int(y2), int(x1): int(x2), :]
                license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(license_plate_crop)
                s = cv2.add(s, 60) 
                hsv_enhanced = cv2.merge([h, s, v])
                image_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

                recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions = detect_text_yolo(
                    ocr_yolo_model, image_enhanced
                )
                print("Image Text ==========================")
                print(detected_numbers, detected_letters)
                print("=====================================")
                print("All Predictions ======================")
                print(all_predictions)
                print("=====================================")

                # Show the image with bounding boxes and predictions
                cv2.imshow("Detected Characters", image_enhanced)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # if license_plate_text is not None:
                #     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                #                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                #                                                     'text': license_plate_text,
                #                                                     'bbox_score': score,
                #                                                     'text_score': license_plate_text_score}}

# # Release video and close windows
# cap.release()
# cv2.destroyAllWindows()

# # Create a DataFrame
# df = pd.DataFrame(excel_data, columns=[
#                   "X1", "Y1", "X2", "Y2", "Character Options"])

# # Save to an Excel file
# df.to_excel("character_predictions.xlsx", index=False)


# write results
write_csv(results, './test.csv')
