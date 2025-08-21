# 🤖 Real-Time Python Chatbot with Flask + TensorFlow

An AI-powered chatbot designed to answer **Python programming questions** in real-time. <br> 
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

real-time-chatbot/<br>
│── static/ # CSS, JS, images<br>
│── templates/ # HTML templates (chat UI)<br>
│── app.py # Flask server<br>
│── chatbot_model.h5 # Trained TensorFlow model<br>
│── chatbot_test.postman_collection.json # Postman API test collection<br>
│── classes.pkl # Pickled class labels<br>
│── intents.json # Training dataset (Python Q&A intents)<br>
│── model.py # Model loading & prediction<br>
│── requirements.txt # Dependencies<br>
│── train_model.py # Training script<br>
│── utils.py # Text preprocessing helpers<br>
│── words.pkl # Pickled vocabulary<br>


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

