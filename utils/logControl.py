import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import datetime

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
    status: str  # 'Denied' or 'Accepted'
):
    log_data = {
        "plateText": plate_text,
        "carDetection": [car_box],       # لازم تكون list of lists
        "plateDetection": [plate_box],   # نفس الكلام
        "user": ObjectId(user_id),
        "garage": ObjectId(garage_id),
        "action": status,
        "accessTime": datetime.utcnow()
    }


    result = logs_col.insert_one(log_data)
    return str(result.inserted_id)

