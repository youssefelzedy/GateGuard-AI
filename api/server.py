from fastapi import FastAPI, WebSocket
import base64
from core.main_V3 import final_model
import cv2
import numpy as np
     
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connection open")

    try:
        while True:
            # Receive Base64-encoded image
            image_base64 = await websocket.receive_text()

            # Decode Base64 to image bytes
            image_bytes = base64.b64decode(image_base64)

            # Convert bytes to a NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image using OpenCV
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text("Error: Invalid image format.")
                continue
            
            # Process image using your YOLO model
            license_plate_text = final_model(frame)
            print("Detected Plate:", license_plate_text)
            
            # Send detected license plate text back
            await websocket.send_text(f"Detected Plate: {license_plate_text}")

    except Exception as e:
        print("Error:", e)
    finally:
        print("Connection closed")