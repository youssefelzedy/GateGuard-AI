from ultralytics import YOLO
import cv2

from sort.sort import *
from utils.util_V1 import get_car, read_license_plate, write_csv, detect_text_yolo, detect_plate_text, crop_LowerPart_Plate
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
cap = cv2.VideoCapture('./Input-videos/test4.mp4')

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
        # print("detections ==========================")
        # print(detections_)
        # print("=====================================")

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector2(frame)[0]
        # print("license_plates ======================")
        # print(license_plates)
        # print("=====================================")
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(
                    y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(
                    license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 128, 255, cv2.THRESH_BINARY_INV)

                # cv2.imwrite('license_plate_threshold.jpg',
                #             license_plate_crop)
                # print("Image saved as license_plate_threshold.jpg")

                # read license plate number
                # license_plate_text, license_plate_text_score = read_license_plate(
                #     license_plate_crop_thresh)

                # height11, width11, _ = license_plate_crop.shape

                # y_start = int(0.37 * height11)

                # # Crop the image (keep the bottom 73%)
                # cropped_image = license_plate_crop[y_start:, :]
                # # Convert to grayscale (Laplacian works best in grayscale)
                # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # # Apply Laplacian filter to enhance edges
                # laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # Compute Laplacian
                # laplacian = np.uint8(np.absolute(laplacian))  # Convert back to uint8

                recognized_image, detected_numbers, detected_letters, character_bboxes, all_predictions = detect_text_yolo(
                    ocr_yolo_model, license_plate_crop
                )
                print("Image Text ==========================")
                print(detected_numbers, detected_letters)
                print("=====================================")
                # display_predictions(recognized_image, character_bboxes, all_predictions)

                # for i, (cx1, cy1, cx2, cy2) in enumerate(character_bboxes):
                #     # Adjust coordinates relative to the cropped license plate image
                #     cropped_character = license_plate_crop[int(
                #         cy1-5):int(cy2+5), int(cx1-5):int(cx2+5)]

                #     # Convert to grayscale (improves OCR accuracy)
                #     gray_char = cv2.cvtColor(
                #         cropped_character, cv2.COLOR_BGR2GRAY)
                #     blurred = cv2.GaussianBlur(gray_char, (3, 3), 0)
                #     thresholded = cv2.adaptiveThreshold(
                #         gray_char,
                #         255,
                #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                #         cv2.THRESH_BINARY,
                #         61, 2  # Try increasing blockSize to 11 or 15
                #     )
                #     # Apply Laplacian filter to enhance edges
                #     laplacian = cv2.Laplacian(
                #         blurred, cv2.CV_64F)  # Compute Laplacian
                #     # Convert back to uint8
                #     laplacian = np.uint8(np.absolute(laplacian))

                #     # Save character image for debugging (optional)
                #     cv2.imwrite(f"cropped_char_{i}.png", thresholded)

                #     texts, boxes = detect_plate_text(thresholded)

                #     print(f"Character {i}: {texts}")

                # print("Image Text ===========gggggg===============")
                # print(texts)
                # print("========================gggggggggggg=============")

                # print(character_bboxes)
                # Show the image with bounding boxes and predictions
                cv2.imshow("Detected Characters", recognized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # # Create an empty list to store all data
                # excel_data = []

                # # Draw bounding boxes for each detected character
                # for (char, bbox) in zip(detected_numbers + detected_letters, character_bboxes):
                #     cx1, cy1, cx2, cy2 = bbox
                #     # Adjust coordinates relative to full image
                #     abs_x1 = int(x1 + cx1)
                #     abs_y1 = int(y1 + cy1)
                #     abs_x2 = int(x1 + cx2)
                #     abs_y2 = int(y1 + cy2)

                #     # Draw rectangle around character
                #     cv2.rectangle(frame, (abs_x1, abs_y1),
                #                   (abs_x2, abs_y2), (0, 0, 255), 4)

                #     # Put detected character near the box
                #     cv2.putText(frame, char, (abs_x1, abs_y1 - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Add data to the Excel sheet
                # for pred in all_predictions:
                #     bbox_x1, bbox_y1, bbox_x2, bbox_y2, character_options = pred
                #     excel_data.append(
                #         [bbox_x1, bbox_y1, bbox_x2, bbox_y2, character_options])

                # # Display the frame
                # cv2.imshow("License Plate Detection", frame)

                # # Press 'q' to quit
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

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
