import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION")
GARAGE_COLLECTION = os.getenv("GARAGE_COLLECTION")
CAMERA_COLLECTION = os.getenv("CAMERA_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
logs_col = db[LOGS_COLLECTION]
garage_col = db[GARAGE_COLLECTION]
camera_col = db[CAMERA_COLLECTION]


def get_active_cameras():
    """
    Retrieve all active cameras from the database.
    Returns:
        list: A list of active cameras.
    """
    cameras = list(camera_col.find({"cameraStatus": "active"}))
    return cameras
