from datetime import datetime, timezone
from flask_bcrypt import Bcrypt
from db import mongo
from bson import ObjectId

bcrypt = Bcrypt()
users_collection = mongo.db.users

def create_user(email, password, name, username):
    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "email": email,
        "password": hashed_pw,
        "name": name,
        "username": username,
        "chat_history": [],  # limited to 10 most recent entries
        "created_at": datetime.now(timezone.utc)
    }
    return users_collection.insert_one(user)

def get_user_by_id(user_id):
    try:
        return users_collection.find_one({ "_id": ObjectId(user_id) })
    except:
        return None

def get_user_by_email(email):
    try:
        return users_collection.find_one({ "email": email }) 
    except:   
        return None

# Why do we use get user_by_id when we have get_user_by_email ?
# This is because user_id is indexed in mongoDB. O(logn) time to get user_by_id. O(n) time to get_user_by_email.
    
def verify_user(email, password):
    user = get_user_by_email(email)
    if user and bcrypt.check_password_hash(user['password'], password):
        return user
    else:
        return None

def update_user_profile(user_id, name, username):
    return users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"name": name, "username": username}}
    )

def update_chat_history(user_id, new_entry):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return False

        history = user.get("chat_history", [])
        history.append(new_entry)
        history = history[-10:]  # Keep only last 10 entries

        users_collection.update_one(
            { "_id": ObjectId(user_id) },
            { "$set": { "chat_history": history } }
        )
        return True
    except Exception as e:
        print("Chat history update failed:", e)
        return False
