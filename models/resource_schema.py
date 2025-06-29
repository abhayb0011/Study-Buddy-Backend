from db import mongo

def add_resource(subject, playlists, books, suggestions):
    mongo.db.resources.insert_one({
        "subject": subject,
        "playlists": playlists,         # list of { title, url }
        "books": books,                 # list of { title, author, url (optional) }
        "suggestions": suggestions      # list of strings
    })

def get_all_resources():
    return list(mongo.db.resources.find({}, {'_id': 0}))
