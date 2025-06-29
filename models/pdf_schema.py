from db import mongo
from datetime import datetime
from bson import ObjectId

def save_pdf_chunks(user_id, chunks):
    pdf_id = str(ObjectId())
    pdf_doc = {
        "_id": pdf_id,
        "user_id": str(user_id),
        "chunks": chunks,
        "created_at": datetime.utcnow()
    }
    mongo.db.pdfs.insert_one(pdf_doc)
    return pdf_id

def get_pdf_chunks(pdf_id):
    pdf_doc = mongo.db.pdfs.find_one({"_id": pdf_id})
    return pdf_doc["chunks"] if pdf_doc else []
