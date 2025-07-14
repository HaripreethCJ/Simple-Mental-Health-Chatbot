from transformers import pipeline
from textblob import TextBlob

# Load chatbot model
chatbot = pipeline("text-generation", model="distilgpt2")

# Detect emotion
def detect_emotion(user_input):
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return "very positive"
    elif polarity > 0:
        return "positive"
    elif polarity == 0:
        return "neutral"
    elif polarity > -0.5:
        return "negative"
    else:
        return "very negative"

# Generate response
def generate_response(user_input):
    emotion = detect_emotion(user_input)

    if emotion == "very negative":
        return "I'm really sorry you're feeling this way. It's okay to feel low sometimes. Please remember, you're not alone. Take deep breaths. 💙"

    elif emotion == "negative":
        return "That sounds tough. Want to talk more about it? I'm here to listen. 🌧️"

    elif emotion == "neutral":
        # ✅ Shorter GPT response only used for neutral emotion
        prompt = f"The user says: {user_input}\nRespond kindly:"
        response = chatbot(
            prompt,
            max_length=60,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=50256  # To prevent warnings
        )[0]['generated_text']
        return response[len(prompt):].strip()

    elif emotion == "positive":
        return "That's great to hear! Small victories lead to big wins. Keep shining ✨"

    elif emotion == "very positive":
        return "Amazing! I'm so proud of your energy. Let your joy inspire others 🔥"

    else:
        return "I'm here for you, no matter how you're feeling. 🤍"





# ✅ THIS PART RUNS THE BOT
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("AI: Take care. You're not alone. 🤍")
        break
    print("AI:", generate_response(user_input))

