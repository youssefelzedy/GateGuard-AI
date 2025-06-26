import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import datetime, timedelta

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION")


client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
logs_col = db[LOGS_COLLECTION]

def create_log(
    plate_text: str,
    car_box: list,
    plate_box: list,
    user_id: str,
    garage_id: str,
    status: str,
    processed: bool = False
):
    """
    Create a log entry in the database.
    Args:
        plate_text (str): The recognized text from the license plate.
        car_box (list): Bounding box coordinates for the detected car.
        plate_box (list): Bounding box coordinates for the detected license plate.
        user_id (str): The ID of the user associated with the garage.
        garage_id (str): The ID of the garage where the log is being created.
        status (str): The action status, e.g., 'Accepted' or 'Denied'.
        Returns:
        str: The ID of the created log entry."""
    
    if user_id is None:
        log_data = {
            "plateText": plate_text,
            "carDetection": [car_box],
            "plateDetection": [plate_box],
            "user": None,
            "garage": ObjectId(garage_id),
            "action": status,
            "accessTime": datetime.utcnow(),
            "processed": processed
        }
    else:
        log_data = {
            "plateText": plate_text,
            "carDetection": [car_box],
            "plateDetection": [plate_box],
            "user": ObjectId(user_id),
            "garage": ObjectId(garage_id),
            "action": status,
            "accessTime": datetime.utcnow(),
            "processed": processed
        }


    result = logs_col.insert_one(log_data)
    return str(result.inserted_id)

def check_log_exists_recently(plate_text: str, cooldown_minutes: int = 12) -> bool:
    """
    Check if a log exists for the same plate text within the last X minutes.
    """
    now = datetime.utcnow()
    cooldown_start = now - timedelta(minutes=cooldown_minutes)

    log = logs_col.find_one({
        "plateText": plate_text,
        "accessTime": {
            "$gte": cooldown_start,
            "$lte": now
        }
    })

    return log is not None