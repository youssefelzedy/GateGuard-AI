from fastapi import FastAPI, WebSocket
import base64
from core.main import final_model

import cv2
import numpy as np
import json 
     
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connection Open...")

    try:
        while True:
            # Receive data
            data = await websocket.receive_text()
            
            # Check if it's a test message
            if data == "CONNECTION_TEST":
                print("Received connection test message")
                await websocket.send_text("Connection successful")
                continue
            
            print(f"Received data of length: {len(data)}")
            
            try:
                # Decode Base64 to image bytes
                image_bytes = base64.b64decode(data)
                
                # Log bytes info
                print(f"Decoded {len(image_bytes)} bytes")
                
                # Convert bytes to a NumPy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Decode image using OpenCV
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("Failed to decode image")
                    # Debug: save the raw bytes to analyze
                    with open("debug_image.bin", "wb") as f:
                        f.write(image_bytes)
                    await websocket.send_text("Error: Invalid image format.")
                    continue
                
                # Debug info
                print(f"Successfully decoded image: {frame.shape}")
                # Process image using your YOLO model
                result = final_model(frame)
                result_json = json.dumps(result)
                
                # Send detected license plate text back
                await websocket.send_text(result_json)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                await websocket.send_text(f"Error: {str(e)}")

    except Exception as e:
        print(f"Connection Error: {str(e)}")
    finally:
        print("Connection Closed...")