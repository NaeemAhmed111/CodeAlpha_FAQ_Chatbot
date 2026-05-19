import string
import streamlit as str_ui
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Knowledge Base
faq_data = {
    "What is your return policy? refundable items": "You can return any product within 30 days of purchase for a full refund.",
    "How long does shipping take? delivery time days": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.",
    "Do you ship internationally? global delivery countries": "Yes, we ship to over 50 countries worldwide. Rates vary by location.",
    "How can I track my order? tracking link status": "Once shipped, you will receive an email with a tracking link to follow your package.",
    "What payment methods do you accept? card paypal cash": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."
}
faq_questions = list(faq_data.keys())

# 2. NLP Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

preprocessed_faqs = [preprocess_text(q) for q in faq_questions]

def get_bot_response(user_query):
    cleaned_query = preprocess_text(user_query)
    if not cleaned_query.strip():
        return "I'm sorry, I didn't quite catch that. Could you rephrase?"
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_faqs + [cleaned_query])
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx][0]
    
    if highest_score < 0.15:
        return "I'm not entirely sure about that. Would you like me to connect you with our support team?"
        
    return faq_data[faq_questions[best_match_idx]]

# 3. Streamlit Premium UI Setup
str_ui.set_page_config(page_title="Smart AI Assistant", page_icon="💬", layout="centered")

# --- Custom CSS for Catchy & Handsome Interface ---
str_ui.markdown("""
    <style>
    /* Gradient Background for the App */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Title and text colors */
    h1 {
        color: #38bdf8 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-align: center;
        text-shadow: 0px 4px 10px rgba(56, 189, 248, 0.2);
    }
    
    p {
        color: #cbd5e1 !important;
        font-size: 1.1rem;
    }
    
    /* Custom Styling for Chat Messages */
    .stChatMessage {
        border-radius: 15px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* User Message Style (Right side vibe) */
    [data-testid="stChatMessageUser"] {
        background-color: #1e293b !important;
        border-left: 5px solid #38bdf8 !important;
    }
    
    /* Bot Message Style */
    [data-testid="stChatMessageAssistant"] {
        background-color: #0f172a !important;
        border-left: 5px solid #10b981 !important;
    }
    
    /* Input Box styling */
    .stChatInputContainer {
        border-radius: 30px !important;
        border: 1px solid #38bdf8 !important;
        background-color: #1e293b !important;
        box-shadow: 0px 4px 20px rgba(56, 189, 248, 0.1) !important;
    }
    
    /* Divider Customization */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #38bdf8, transparent);
        margin: 25px 0;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
str_ui.markdown("<h1>⚡ NOVA • Smart FAQ Assistant</h1>", unsafe_allow_html=True)
str_ui.markdown("<p style='text-align: center;'>Ask your questions below. Powered by NLP & Cosine Similarity.</p>", unsafe_allow_html=True)
str_ui.markdown("<hr>", unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in str_ui.session_state:
    str_ui.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Nova, your virtual assistant. Ask me anything about payments, shipping, or returns!"}
    ]

# Display past chat messages
for message in str_ui.session_state.messages:
    with str_ui.chat_message(message["role"]):
        str_ui.write(message["content"])

# User Input Box
if user_input := str_ui.chat_input("Type your message here..."):
    # Display user message
    with str_ui.chat_message("user"):
        str_ui.write(user_input)
    str_ui.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate and display bot response
    bot_response = get_bot_response(user_input)
    with str_ui.chat_message("assistant"):
        str_ui.write(bot_response)
    str_ui.session_state.messages.append({"role": "assistant", "content": bot_response})