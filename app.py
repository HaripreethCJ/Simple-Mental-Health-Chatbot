import gradio as gr
from transformers import pipeline
from textblob import TextBlob

print("🚀 Launching Mental Health Chatbot App...")

# Load chatbot model
chatbot = pipeline("text-generation", model="distilgpt2")

# Emotion detection logic
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

# Chatbot logic
def generate_response(user_input):
    emotion = detect_emotion(user_input)

    if emotion == "very negative":
        return "I'm really sorry you're feeling this way. You're not alone. 💙"
    elif emotion == "negative":
        return "That sounds tough. I'm here to listen. 🌧️"
    elif emotion == "neutral":
        prompt = f"The user says: {user_input}\nRespond kindly:"
        response = chatbot(
            prompt,
            max_length=60,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=50256
        )[0]['generated_text']
        return response[len(prompt):].strip()
    elif emotion == "positive":
        return "That's great to hear! Keep shining ✨"
    elif emotion == "very positive":
        return "Amazing! Your joy is inspiring 🔥"
    else:
        return "I'm here for you, always 🤍"

# Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="How are you feeling today?"),
    outputs="text",
    title="🧠 Mental Health Chatbot",
    description="An emotionally aware chatbot that responds with empathy."
)

# ✅ This launches the web app
interface.launch()

