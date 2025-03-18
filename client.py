import cv2
import base64
import websockets
import asyncio

async def send_image():
    image_path = "image.jpg"  # Change to your image path
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Unable to load image.")
        return

    _, buffer = cv2.imencode('.jpg', frame)  # Convert to JPEG format
    image_bytes = buffer.tobytes()  # Convert to bytes
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  # Encode in Base64

    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        await websocket.send(image_base64)  # Send image data as a Base64 string
        response = await websocket.recv()  # Wait for server response
        print("Server response:", response)

# Run WebSocket client
asyncio.run(send_image())
