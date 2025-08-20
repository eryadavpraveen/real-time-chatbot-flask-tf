from flask import Flask, request, jsonify, render_template
from model import predict_class, get_response
import json

app = Flask(__name__)

# Load intents.json once
with open("intents.json", encoding="utf-8") as f:
    intents_json = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    print("User said:", user_message)

    try:
        ints = predict_class(user_message)
        res = get_response(ints, intents_json)
    except Exception as e:
        res = f"Error: {str(e)}"

    print("Bot replied:", res)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)
