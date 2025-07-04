import streamlit as st
from dataclasses import dataclass
from typing import Literal
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Constants
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "code": "en"},
    "ms": {"name": "Bahasa Melayu", "code": "ms"},
    "bn": {"name": "বাংলা (Bengali)", "code": "bn"},
    "fr": {"name": "Français", "code": "fr"},
    "es": {"name": "Español", "code": "es"},
    "de": {"name": "Deutsch", "code": "de"},
    "ja": {"name": "日本語 (Japanese)", "code": "ja"}
}

# Message structure
@dataclass
class Message:
    origin: Literal["user", "bot"]
    text: str

# Initialize session state
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

# Load QA model and corpus
@st.cache_resource
def load_models_and_data():
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model_name = "deepset/roberta-base-squad2"
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    with open("data/wiki_summaries.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    corpus = []
    for item in data:
        title = item.get("title", "")
        summary = item.get("summary", "")
        # Skip items without title or summary
        if not title or not summary:
            continue
            
        documents.append({"title": title, "context": summary})
        corpus.append(title + " " + summary)

    corpus_embeddings = retriever.encode(corpus, convert_to_tensor=True)
    
    return retriever, qa_model, qa_tokenizer, documents, corpus_embeddings

# Utility functions
def answer_question(question, context, qa_model, qa_tokenizer):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_ids = inputs["input_ids"][0][start_idx:end_idx]
    return qa_tokenizer.decode(answer_ids, skip_special_tokens=True)

def translate(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        return f"[Translation Error: {str(e)}]"

def detect_language(text):
    try:
        detected = detect(text)
        # Normalize common variants (e.g. Malay/Indonesian)
        if detected in ["id", "may", "ms"]:
            detected = "ms"
        return detected if detected in SUPPORTED_LANGUAGES else "en"
    except:
        return "en"

# UI functions
def apply_custom_css():
    st.markdown("""
        <style>
            .user-msg {
                background-color: #DCF8C6;
                color: black;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 5px;
                text-align: right;
            }
            .bot-msg {
                background-color: #EAEAEA;
                color: black;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 5px;
                text-align: left;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.5em 1em;
                margin: 0.3em 0.3em;
            }
            .clear-button > button {
                background-color: #f44336 !important;
                color: white !important;
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)

def display_chat_history():
    for msg in st.session_state.chat_history:
        if msg.origin == "user":
            st.markdown(f"<div class='user-msg'>{msg.text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg.text}</div>", unsafe_allow_html=True)

def create_language_selector():
    lang_options = list(SUPPORTED_LANGUAGES.keys())
    format_func = lambda x: SUPPORTED_LANGUAGES[x]["name"]
    
    return st.radio(
        "Output language",
        options=lang_options,
        format_func=format_func,
        horizontal=True,
        index=lang_options.index(st.session_state["lang"])
    )

def create_sidebar(documents):
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This multilingual QA chatbot can:
        - **Auto-detect** your input as per the supported language
        - **Answer questions** from the knowledge base
        - **Translate responses** to your preferred language
        """)
        
        st.header("Supported Languages")
        for code, info in SUPPORTED_LANGUAGES.items():
            st.markdown(f"• {info['name']} ({code})")
        
        st.header("Statistics")
        st.metric("Total Documents", len(documents))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        if st.button("Reload Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

def process_user_input(user_input, selected_lang, retriever, qa_model, qa_tokenizer, documents, corpus_embeddings):
    # Add user message
    st.session_state.chat_history.append(Message("user", user_input.strip()))

    # Detect and normalize language
    detected_lang = detect_language(user_input.strip())

    # Translate question to English for processing
    query_en = (
        user_input.strip() if detected_lang == "en"
        else translate(user_input.strip(), detected_lang, "en")
    )

    # Retrieve top document
    query_embedding = retriever.encode(query_en, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0]

    # Generate and translate answer
    for hit in hits:
        doc = documents[hit["corpus_id"]]
        answer_en = answer_question(query_en, doc["context"], qa_model, qa_tokenizer)

        # Translate answer to selected language
        final_answer = (
            answer_en if selected_lang == "en"
            else translate(answer_en, "en", selected_lang)
        )

        # Add bot response
        st.session_state.chat_history.append(
            Message("bot", f"From: {doc['title']}<br>Answer: {final_answer}")
        )

# Main application
def main():
    # Initialize
    init_session_state()
    apply_custom_css()
    
    # Load models and data
    retriever, qa_model, qa_tokenizer, documents, corpus_embeddings = load_models_and_data()
    
    # Title
    st.title("🗨️ Multilingual QA Chatbot")
    
    # Display chat history
    display_chat_history()
    
    # Language selection
    st.session_state["lang"] = create_language_selector()
    
    # Input form
    st.subheader("✍️ Ask a Question")
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your question in any supported language (Check Sidebar for info):",
            placeholder="Example: What is artificial intelligence?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([5, 1])
        with col1:
            submitted = st.form_submit_button("Submit")
        with col2:
            clear = st.form_submit_button("Clear Chat")
    
    # Handle buttons
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submitted and user_input.strip():
        process_user_input(
            user_input, 
            st.session_state["lang"], 
            retriever, 
            qa_model, 
            qa_tokenizer, 
            documents, 
            corpus_embeddings
        )
        st.rerun()
    
    # Sidebar
    create_sidebar(documents)

if __name__ == "__main__":
    main()