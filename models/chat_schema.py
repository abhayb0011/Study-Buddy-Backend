from db import mongo
from datetime import datetime
from bson import ObjectId

chats_collection = mongo.db.chats

# Save a single chat interaction
def save_chat(user_id, question, answer, source="pdf"):
    chat_doc = {
        "user_id": ObjectId(user_id),
        "question": question,
        "answer": answer,
        "source": source,  # could be 'pdf', 'planner', etc.
        "timestamp": datetime.utcnow()
    }
    chats_collection.insert_one(chat_doc)

# Get last N chats of a user
def get_recent_chats(user_id, limit=10):
    return list(
        chats_collection.find({"user_id": ObjectId(user_id)})
        .sort("timestamp", -1)
        .limit(limit)
    )