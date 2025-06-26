from fastapi import FastAPI, Response
import base64
import traceback

from core.main import final_model
from utils.cameraControl import get_active_cameras
from utils.userControl import get_users_by_garage
from utils.logControl import create_log, check_log_exists_recently

import cv2
import numpy as np
import json
import threading
import time

app = FastAPI()

latest_frame = None
latest_result = None
latest_result_json = None

def process_frames():
    """
    Continuously capture frames from the IP camera, process them with the YOLO model,
    and log the results.
    This function runs in a separate thread to avoid blocking the main application.
    It retrieves active cameras from the database, captures frames, decodes them,
    and processes them to detect license plates and vehicles.
    It also logs the results in the database.
    """
    global latest_frame, latest_result, latest_result_json

    cameras = get_active_cameras()

    if not cameras:
        print("No active cameras found.")
        return
    
    while True:
        for camera in cameras:
            print(f"Camera ID: {camera['cameraIP']}, Name: {camera['garage']}")

            # IP Camera URL (change this to your actual IP camera stream)
            CAMERA_URL = camera['cameraIP']

            cap = cv2.VideoCapture(CAMERA_URL)
            if not cap.isOpened():
                print("Camera connection failed.")
                continue

            try:
                ret, frame = cap.read()
                if ret:
                    _, jpeg = cv2.imencode('.jpg', frame)
                    latest_frame = jpeg.tobytes()

                    try:
                        # Decode Base64 to image bytes
                        # image_bytes = base64.b64decode(latest_frame)
                        

                        # Log bytes info
                        print(f"Decoded {len(latest_frame)} bytes")
                        
                        # Convert bytes to a NumPy array
                        nparr = np.frombuffer(latest_frame, np.uint8)
                        
                        # Decode image using OpenCV
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("Failed to decode image")
                            # Debug: save the raw bytes to analyze
                            latest_result = "Error: Invalid image format."
                            latest_result_json = json.dumps({"error": "Invalid image format"})
                            continue
                        
                        # Debug info
                        print(f"Successfully decoded image: {frame.shape}")

                        # Process image using your YOLO model
                        latest_result = final_model(frame)
                        latest_result_json = json.dumps(latest_result)

                        for plate in latest_result:

                            if not isinstance(plate, list) or len(plate) < 3:
                                print("Invalid plate detection format")
                                continue

                            result_of_search = get_users_by_garage(camera['garage'], plate[0])

                            if result_of_search:
                                # check the log for the user in the garage
                                logs = check_log_exists_recently(plate[0])
                                print(f"User found: {result_of_search}")

                                if not logs:
                                    log = create_log(
                                        plate_text=plate[0],
                                        car_box=plate[1],
                                        plate_box=plate[2],
                                        user_id=result_of_search['_id'],
                                        garage_id=camera['garage'],
                                        status='Accepted'
                                    )
                                    print(f"User found: {log}")
                            else:
                                # check the log for the user in the garage
                                logs = check_log_exists_recently(plate[0])
                                print(f"Detected plate: {plate[0]}")

                                if not logs:
                                    log = create_log(
                                        plate_text=plate[0],
                                        car_box=plate[1],
                                        plate_box=plate[2],
                                        user_id=None,
                                        garage_id=camera['garage'],
                                        status='Denied'
                                    )
                                    print(f"No user found for this plate. {log}")                                   
                        
                    except Exception as e:
                        print("Error processing frame:")
                        traceback.print_exc()
                        latest_result = (f"Error: {str(e)}")
                        latest_result_json = json.dumps({"error": str(e)})

            except Exception as e:
                print(f"Connection Error: {str(e)}")
            finally:
                print("Connection Closed...")

        # Wait for a short period before the next iteration
        time.sleep(12)

 

# Start thread to capture frames from the IP camera.
threading.Thread(target=process_frames, daemon=True).start()


# @app.get("/frame")
# def get_frame():
#     if latest_frame is None:
#         return Response(content="No frame yet", status_code=503)
#     return Response(content=latest_frame, media_type="image/jpeg")

@app.get("/result")
def get_result():
    """
    Retrieve the latest processed result.
    Returns:
        dict: The latest result in JSON format, or a message indicating no result yet.
    """
    if latest_result_json is None:
        return {"message": "No result yet"}
    return latest_result_json

