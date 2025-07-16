from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from pymongo.errors import ConnectionFailure

# Create mongo object of PyMongo class. This is used to interact with MongoDB.
mongo = PyMongo()

# Load the environment variables from .env file
load_dotenv()

def init_db(app):
    app.config["MONGO_URI"] = os.getenv("MONGO_URI") # Set MongoDB URI for connection 
    mongo.init_app(app) # Initialize mongo with the Flask app. Now mongo is tied to our Flask app and can be used to acces the database.

def test_connection():
    try:
        # Attempt to connect to the MongoDB server
        mongo.cx.admin.command('ping')
        print("MongoDB connection successful.")
    except ConnectionFailure:
        print("MongoDB connection failed.")