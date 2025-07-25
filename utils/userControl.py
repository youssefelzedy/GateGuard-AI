import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USER_COLLECTION = os.getenv("USER_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
users_col = db[USER_COLLECTION]

def get_users_by_garage(garage_id: str, car_plate: str = None):
    """
    Retrieve users associated with a specific garage and optionally filter by car plate.
    Args:
        garage_id (str): The ID of the garage.
        car_plate (str, optional): The car plate to filter users by. Defaults to None.
    Returns:
        dict: The user document if found, otherwise None.
    """
    user = users_col.find_one({
        "carPlate": car_plate,
        "garage": ObjectId(garage_id)
    })

    if not user:
        return None

    return user