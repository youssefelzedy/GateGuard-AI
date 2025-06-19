import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
GARAGE_COLLECTION = os.getenv("GARAGE_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
garages_col = db[GARAGE_COLLECTION]

def check_garage_status(garage_id: str):
    """
    Check the status of a garage.
    Args:
        garage_id (str): The ID of the garage to check.
    Returns:
        dict: The current status of the garage.
    """
    garage = garages_col.find_one({"_id": ObjectId(garage_id)})
    if garage:
        return {"open": garage.get("status", False)}
    return {"message": "Garage not found"}

def update_garage_status(garage_id: str, status: bool):
    """
    Update the status of a garage.
    Args:
        garage_id (str): The ID of the garage to update.
        status (bool): The new status of the garage (open/closed).
    Returns:
        dict: The result of the update operation.
    """
    result = garages_col.update_one({"_id": ObjectId(garage_id)}, {"$set": {"status": status}})
    if result.modified_count > 0:
        return {"message": "Garage status updated successfully"}
    return {"message": "Garage not found or status unchanged"}