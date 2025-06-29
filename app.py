import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from db import mongo, test_connection
import jwt
from datetime import datetime, timedelta, timezone
from models.user_schema import create_user, get_user_by_id, get_user_by_email, verify_user, update_chat_history
from models.chat_schema import save_chat  
from models.pdf_schema import save_pdf_chunks, get_pdf_chunks
from models.resource_schema import get_all_resources
import pdfplumber
import pickle
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from collections import Counter
import google.generativeai as genai
from waitress import serve

# Load env vars
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('models/gemini-1.5-flash')

# Load model & tokenizer
model_path = os.path.join(os.getcwd(), 'bert_model')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB URI config
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/myDatabase")

# Initialize mongo
mongo.init_app(app)
test_connection(app)

# JWT secret
SECRET_KEY = os.getenv("JWT_SECRET", "supersecretkey")

# --- JWT utility ---
def generate_token(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# --- Auth Decorator ---
def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace("Bearer ", "")
        if not token:
            return jsonify({'error': 'Missing token'}), 401

        user_id = decode_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 403

        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# Helper Functions
def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_chunks(query, chunks, top_k=3):
    try:
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']

        scores = []
        for chunk in chunks:
            chunk_embedding = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )['embedding']

            similarity = cosine_similarity(query_embedding, chunk_embedding)
            scores.append((similarity, chunk))

        scores.sort(reverse=True)
        top_chunks = [chunk for _, chunk in scores[:top_k]]
        return "\n".join(top_chunks)

    except Exception as e:
        print("Embedding error:", e)
        return ""

def predict_topic(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    preds = tf.argmax(logits, axis=1).numpy()
    return le.inverse_transform(preds)

def extract_questions_from_text(text):
    prompt = f"""
You are an expert at extracting multiple choice questions from academic content.
Extract all individual multiple choice questions from the text below.
Only return a clean Python list of strings.
Do NOT include explanations or numbers. No markdown. No extra formatting.

Text:
'''{text}'''
Return the list of questions only.
"""
    try:
        response = model_gemini.generate_content(prompt)
        return eval(response.text.strip())
    except Exception as e:
        print("Gemini error:", e)
        return []

def retrieve_chunks_from_session_or_db(pdf_id):
    return get_pdf_chunks(pdf_id)

# --- Routes ---

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    username = data.get('username')

    if get_user_by_email(email):
        return jsonify({'error': 'User already exists'}), 409

    create_user(email, password, name, username)
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = verify_user(email, password)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    token = generate_token(user['_id'])
    return jsonify({'token': token}), 200

@app.route('/profile', methods=['GET'])
@require_auth
def profile(user):
    return jsonify({
        'email': user['email'],
        'name': user['name'],
        'username': user['username'],
        'chat_history': user.get('chat_history', []),
        'created_at': user['created_at']
    }), 200

@app.route("/quote-of-the-day", methods=["GET"])
def quote_of_the_day():
    try:
        response = requests.get("https://zenquotes.io/api/today")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch quote", "details": str(e)}), 500

@app.route('/upload-pyq', methods=['POST'])
@require_auth
def upload_pyq(user):
    file = request.files.get('pdf')
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid or missing PDF file'}), 400

    try:
        with pdfplumber.open(file) as pdf:
            full_text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        return jsonify({'error': f'PDF reading failed: {str(e)}'}), 500

    if not full_text.strip():
        return jsonify({'error': 'No text found in PDF'}), 400

    questions = extract_questions_from_text(full_text)
    if not questions:
        return jsonify({'error': 'No questions extracted'}), 400

    predicted_topics = predict_topic(questions)
    topic_counts = Counter(predicted_topics)
    total = sum(topic_counts.values())
    topic_percentages = {topic: f"{(count / total) * 100:.2f}%" for topic, count in topic_counts.items()}

    summary = f"Uploaded PYQ - Top topic: {topic_counts.most_common(1)[0][0]}"
    update_chat_history(user['_id'], summary)

    return jsonify({
        "topic_counts": dict(topic_counts),
        "topic_percentages": topic_percentages
    }), 200

@app.route('/upload-chat-pdf', methods=['POST'])
@require_auth
def upload_chat_pdf(user):
    file = request.files.get('pdf')
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid or missing PDF file'}), 400

    try:
        with pdfplumber.open(file) as pdf:
            full_text = '\n'.join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    except Exception as e:
        return jsonify({'error': f'PDF reading failed: {str(e)}'}), 500

    if not full_text.strip():
        return jsonify({'error': 'No text found in PDF'}), 400

    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
    pdf_id = save_pdf_chunks(user['_id'], chunks)

    return jsonify({"pdf_id": pdf_id, "chunk_count": len(chunks)}), 200

@app.route('/chat-with-pdf', methods=['POST'])
@require_auth
def chat_with_pdf(user):
    question = request.json.get('question')
    pdf_id = request.json.get('pdf_id')

    if not question:
        return jsonify({"error": "Missing question"}), 400

    chunks = retrieve_chunks_from_session_or_db(pdf_id)
    relevant_chunks = find_similar_chunks(question, chunks)

    prompt = f"""
You are helping students study by answering questions from a document.
Based on the content below, answer the user's question concisely.

Document Content:
{relevant_chunks}

User Question:
{question}
"""

    try:
        response = model_gemini.generate_content(prompt)
        answer = response.text.strip()

        #  Save to chat collection
        save_chat(user['_id'], question, answer, source="pdf")

        #  Update limited user chat history (last 10 entries)
        entry = f"Q: {question[:50]}... A: {answer[:50]}..."
        update_chat_history(user['_id'], entry)

        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resources', methods=['GET'])
def get_resources():
    try:
        resources = get_all_resources()
        return jsonify({"resources": resources}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  Run Flask App
if __name__ == '__main__':
    try:
        #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
        serve(app, port=8080, host="0.0.0.0")
    except Exception as e:
        import traceback
        print("Exception in main:")
        traceback.print_exc()
