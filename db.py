from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from pymongo.errors import ConnectionFailure

mongo = PyMongo()

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure MongoDB connection
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/myDatabase")

# Initialize PyMongo
mongo = PyMongo(app)

def test_connection(app):
    try:
        mongo.cx.server_info()  # This will raise exception if not connected
        print("MongoDB connected successfully!")
    except ConnectionFailure:
        print("MongoDB connection failed!")