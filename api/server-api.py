from fastapi import FastAPI, Response
import base64
from core.main import final_model

import cv2
import numpy as np
import json
import threading
import time

app = FastAPI()


# IP Camera URL (change this to your actual IP camera stream)
CAMERA_URL = "http://52.158.32.174:5000/video_feed"


latest_frame = None
latest_result = None
latest_result_json = None

def process_frames():
    global latest_frame, latest_result
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Camera connection failed.")
        return
    try:
        while True:
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
                    print(f"Processed result: {latest_result_json}")
                    
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    latest_result = (f"Error: {str(e)}")
                    latest_result_json = json.dumps({"error": str(e)})

            time.sleep(1)
    except Exception as e:
        print(f"Connection Error: {str(e)}")
    finally:
        print("Connection Closed...")

 
'''
Start thread to capture frames from the IP camera.
'''
threading.Thread(target=process_frames, daemon=True).start()


# @app.get("/frame")
# def get_frame():
#     if latest_frame is None:
#         return Response(content="No frame yet", status_code=503)
#     return Response(content=latest_frame, media_type="image/jpeg")

@app.get("/result")
def get_result():
    if latest_result_json is None:
        return {"message": "No result yet"}
    return latest_result_json
