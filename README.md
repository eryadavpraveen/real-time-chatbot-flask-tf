# 🤖 Real-Time Python Chatbot with Flask + TensorFlow

An AI-powered chatbot designed to answer **Python programming questions** in real-time.  
Built with **Flask, TensorFlow, NLTK, and MongoDB**, this chatbot provides a simple web interface and REST API support.  

---

## 🚀 Features
- Real-time chatbot that answers **Python-specific questions**  
- Built with **Flask + TensorFlow** backend  
- NLP preprocessing using **NLTK**  
- Intent classification trained on custom dataset  
- MongoDB for storing chat history  
- Simple UI with timestamps (like a messaging app)  
- REST API tested via **Postman**  

---

## 🛠️ Tech Stack
- **Backend:** Flask, TensorFlow, Keras  
- **NLP:** NLTK, WordNet Lemmatizer  
- **Database:** MongoDB  
- **Frontend:** HTML, CSS, JavaScript (chat interface)  
- **Tools:** Postman, Git, Docker (optional for deployment)  

---

## 📂 Project Structure

real-time-chatbot/
│── static/ # CSS, JS, images
│── templates/ # HTML templates (chat UI)
│── app.py # Flask server
│── chatbot_model.h5 # Trained TensorFlow model
│── chatbot_test.postman_collection.json # Postman API test collection
│── classes.pkl # Pickled class labels
│── intents.json # Training dataset (Python Q&A intents)
│── model.py # Model loading & prediction
│── requirements.txt # Dependencies
│── train_model.py # Training script
│── utils.py # Text preprocessing helpers
│── words.pkl # Pickled vocabulary


## Create virtual environment & install dependencies
python -m venv env
source env/bin/activate    # For Linux/Mac
env\Scripts\activate       # For Windows

pip install -r requirements.txt


## Train the chatbot model (if needed)
python train_model.py


## Run the Flask app
python app.py

Now visit 👉 http://127.0.0.1:5000 in your browser.

