# 📚 Study Buddy - Backend

The backend of **Study Buddy**, an AI-powered educational assistant that helps students analyze past year questions, upload study material, and ask questions directly from it. Built with **Flask**, **MongoDB**, **Gemini API**, and **Transformers**.

---

## 🚀 Features

- 🔐 **JWT-based Authentication**
- 📄 **Upload PYQs** and extract questions using Gemini
- 🧠 **Topic Prediction** using a fine-tuned BERT model
- 📘 **Upload Study Material & Ask Questions**
- 🧾 **User Profile** with editable name and username
- 🧠 **Recent Chat History** tracking
- 📚 **Learning Resources**

---

## 🛠️ Tech Stack

- **Backend**: Flask + Waitress
- **Database**: MongoDB (with `flask-pymongo`)
- **AI Models**:
  - `DistilBERT` (via Hugging Face Transformers)
  - Google **Gemini 1.5 Flash API**
- **PDF Parsing**: `pdfplumber`
- **Authentication**: JWT (`pyjwt`)
- **Cloud Deployment**: Google Cloud Run
- **Others**: dotenv, CORS, bcrypt, gdown

---

## 📁 Folder Structure

```

📦 backend/
├── models/
│   ├── user\_schema.py
│   ├── chat\_schema.py
│   ├── pdf\_schema.py
│   └── resource\_schema.py
├── bert\_model/
│   ├── t5\_model.h5
│   └── label\_encoder.pkl
├── app.py
├── db.py
├── requirements.txt
└── .env

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/abhayb0011/study-buddy-backend.git
cd study-buddy-backend
````

### 2. Create `.env` file

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/studybuddy
JWT_SECRET=your_secret_key
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

For development:

```bash
python app.py
```

For production:

```bash
waitress-serve --host=0.0.0.0 --port=8080 app:app
```

---

## 🔐 API Endpoints

### 📌 Auth

| Route             | Method | Description              |
| ----------------- | ------ | ------------------------ |
| `/signup`         | POST   | Register new user        |
| `/login`          | POST   | Login user & get token   |
| `/profile`        | GET    | Fetch user profile       |
| `/update-profile` | PUT    | Update name and username |

> 🔒 Protected routes require `Authorization: Bearer <token>` header.

---

## 📄 Upload PYQs

| Route         | Method | Description                         |
| ------------- | ------ | ----------------------------------- |
| `/upload-pyq` | POST   | Upload a PDF of past year questions |

---

## 📘 Upload Study Material

| Route              | Method | Description                            |
| ------------------ | ------ | -------------------------------------- |
| `/upload-chat-pdf` | POST   | Upload any study material as a PDF     |
| `/chat-with-pdf`   | POST   | Ask a question and get answer from PDF |

---

## 📚 Learning Resources

| Route               | Method | Description                   |
| ------------------- | ------ | ----------------------------- |
| `/resources`        | GET    | Get list of curated resources |
| `/quote-of-the-day` | GET    | Daily motivational quote      |

---

## 📦 Models

* `t5_model.h5` – fine-tuned DistilBERT classification model
* `label_encoder.pkl` – maps predicted labels back to topic names

> As t5_model.h5 size is huge so it is present in google drive. It is downloaded from there.

---

## 🙌 Authors

* **Abhay Bhardwaj**
* **Ankit Chaurasiya**

```
