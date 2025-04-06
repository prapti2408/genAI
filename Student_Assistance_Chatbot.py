import os
import json
import requests
import streamlit as st
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import base64
from gtts import gTTS
import io
import google.generativeai as genai
from typing import List, Dict
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings  

from langchain.text_splitter import RecursiveCharacterTextSplitter
st.sidebar.header("Text-to-Speech Settings")
tts_language = st.sidebar.selectbox(
    "Select Language",
    options=[
        "English", "Spanish", "French", "German", 
        "Italian", "Portuguese", "Hindi", "Chinese"
    ],
    key="tts_language"  # This stores the selection in session state
)

# Language code mapping for gTTS
language_codes = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Dutch": "nl",
    "Turkish": "tr",
    "Swedish": "sv",
    "Finnish": "fi",
    "Danish": "da",
    "Polish": "pl",
    "Indonesian": "id",
    "Thai": "th",
    "Vietnamese": "vi"
}

# Load environment variables
load_dotenv()

# Initialize Gemini model
GOOGLE_API_KEY = "YOUR API KEY"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize RAG components
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize session states
def initialize_session_state():
    if 'gemini_response' not in st.session_state:
        st.session_state.gemini_response = ""
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'play_response_audio' not in st.session_state:
        st.session_state.play_response_audio = False
    if 'play_quiz_audio' not in st.session_state:
        st.session_state.play_quiz_audio = False
    if 'play_custom_audio' not in st.session_state:
        st.session_state.play_custom_audio = False
    if 'custom_text' not in st.session_state:
        st.session_state.custom_text = ""
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []
    if 'score' not in st.session_state:
        st.session_state.score = 0

initialize_session_state()

def build_retriever(text):
    """Create a vector store from extracted text"""
    if not text:
        return None
    
    # Split text into chunks
    texts = text_splitter.split_text(text)
    
    # Create vector store
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def rag_retrieve(query, k=3):
    """Retrieve relevant context using RAG"""
    if st.session_state.vector_store is None:
        return ""
    
    docs = st.session_state.vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def get_gemini_response(prompt, context=""):
    """Get response from Gemini with RAG context"""
    try:
        if context:
            prompt = f"""Use the following context to answer the question:
            {context}
            
            Question: {prompt}
            
            Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return "Error fetching response from Gemini API."

# Document processing functions remain the same
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.text for p in soup.find_all('p')])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching webpage: {str(e)}")
        return ""

# Quiz generation with RAG enhancement
def generate_quiz_questions(content, num_questions=5):
    """Generate quiz questions with robust JSON parsing"""
    if not content:
        return []

    prompt = f"""Generate exactly {num_questions} multiple choice quiz questions based on this content:
    {content[:5000]}  # Truncate to avoid token limits
    
    Format each question as:
    {{
        "question": "text",
        "options": ["a) Option 1", "b) Option 2", "c) Option 3", "d) Option 4"],
        "answer": "b) Option 2"
    }}
    
    Return ONLY a valid JSON array with no Markdown formatting, like this:
    [
        {{...question 1...}},
        {{...question 2...}}
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            return []

        # Clean the response
        response_text = response.text.strip()
        
        # Remove Markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
        
        # Parse with error handling
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list):
                return questions[:num_questions]
            return []
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse quiz JSON: {str(e)}")
            st.text("Raw API Response:")
            st.text(response_text)  # Debug output
            return []
            
    except Exception as e:
        st.error(f"Quiz generation error: {str(e)}")
        return []
# Text-to-speech function remains the same
def text_to_speech(text, lang='en'):
    if not text or len(text.strip()) == 0:
        st.warning("No text to convert to speech.")
        return None
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Callback functions
def on_generate_quiz_click():
    if st.session_state.extracted_text:
        with st.spinner("Generating quiz..."):
            questions = generate_quiz_questions(st.session_state.extracted_text)
            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = [None] * len(questions)
                st.session_state.score = 0
                st.success(f"Generated {len(questions)} quiz questions!")
            else:
                st.error("Failed to generate quiz questions")

def on_listen_response_click():
    st.session_state.play_response_audio = True

def on_listen_quiz_click():
    st.session_state.play_quiz_audio = True

def on_speak_custom_text_click():
    st.session_state.custom_text = st.session_state.tts_text
    st.session_state.play_custom_audio = True

def on_get_answer_click():
    user_input = st.session_state.user_input
    if user_input and st.session_state.vector_store:
        # Retrieve relevant context using RAG
        context = rag_retrieve(user_input)
        # Get response with context
        response = get_gemini_response(user_input, context)
        st.session_state.gemini_response = response
    elif user_input:
        # Fallback to direct Gemini response
        response = get_gemini_response(user_input)
        st.session_state.gemini_response = response

# Streamlit UI
st.title("📚 RAG-Enhanced Student Assistant")

# Sidebar for document upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
webpage_url = st.sidebar.text_input("Enter webpage URL")

# Process uploaded files
if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        extracted_text = extract_text_from_pdf(temp_file_path)
        all_text += extracted_text + "\n\n"
        
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    if all_text:
        st.session_state.extracted_text = all_text
        st.session_state.vector_store = build_retriever(all_text)
        st.text_area("Extracted Text", all_text, height=200)

# Process webpage URL
if webpage_url and st.sidebar.button("Extract Text from URL"):
    extracted_text = extract_text_from_webpage(webpage_url)
    if extracted_text:
        st.session_state.extracted_text = extracted_text
        st.session_state.vector_store = build_retriever(extracted_text)
        st.text_area("Extracted Webpage Content", extracted_text, height=200)

# Main chat interface
st.text_input("Ask something:", key="user_input")
st.button("Get Answer", on_click=on_get_answer_click)

# Display response with RAG context
if st.session_state.gemini_response:
    st.subheader("Response:")
    st.write(st.session_state.gemini_response)
    
    if st.button("🔊 Listen to Response", key="listen_response_button"):
        lang_code = language_codes.get(tts_language, "en")
        with st.spinner("Generating audio..."):
            response_text = st.session_state.gemini_response
            if len(response_text) > 5000:
                response_text = response_text[:5000] + "..."
            audio_bytes = text_to_speech(response_text, lang=lang_code)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

# Quiz section
st.button("Generate Quiz from Content", on_click=on_generate_quiz_click)

if hasattr(st.session_state, 'quiz_questions') and st.session_state.quiz_questions:
    st.subheader("📝 Quiz Time!")
    
    with st.form("quiz_form"):
        if len(st.session_state.user_answers) != len(st.session_state.quiz_questions):
            st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
        
        for i, question in enumerate(st.session_state.quiz_questions):
            st.markdown(f"{i+1}. {question['question']}")
            st.session_state.user_answers[i] = st.radio(
                f"Choose an option for Q{i+1}",
                question["options"],
                key=f"quiz_q{i}"
            )
        
        if st.form_submit_button("Submit Quiz"):
            st.session_state.quiz_submitted = True
            st.session_state.score = sum(
                1 for i, q in enumerate(st.session_state.quiz_questions)
                if st.session_state.user_answers[i] == q["answer"]
            )
            st.rerun()
    
    if st.session_state.quiz_submitted:
        st.success(f"Your score: {st.session_state.score}/{len(st.session_state.quiz_questions)}")
        if st.button("Try Again"):
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
            st.session_state.score = 0
            st.rerun()

# Text-to-speech section
st.subheader("Custom Text-to-Speech")
st.text_area("Enter any text to convert to speech:", height=100, key="tts_text")
st.button("🔊 Speak Text", on_click=on_speak_custom_text_click)

if st.session_state.play_custom_audio and st.session_state.custom_text:
    lang_code = language_codes.get(tts_language, "en")
    with st.spinner("Converting text to speech..."):
        audio_bytes = text_to_speech(st.session_state.custom_text, lang=lang_code)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    st.session_state.play_custom_audio = False
