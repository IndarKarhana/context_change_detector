import streamlit as st
import speech_recognition as sr
from pathlib import Path
import hashlib
import google.generativeai as genai

def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Web Speech API could not understand audio"
        except sr.RequestError as e:
            return "Could not request results from Google Web Speech API; {0}".format(e)

def extract_pdf_pages(pathname: str) -> list[str]:
    parts = [f"--- START OF PDF ${pathname} ---"]
    pages = []
    with open(pathname,'r') as f:
        for index, page in enumerate(f.readlines()):
            parts.append(f"--- PAGE {index} ---")
            parts.append(page)
    return parts

# Streamlit app starts here
st.title("Audio to Text and Conversational AI")

# Input API key
api_key = st.text_input("Enter your Google GenerativeAI API Key")

if st.button("Submit API Key"):
    genai.configure(api_key=api_key)
    st.success("API Key submitted successfully!")

# File upload section
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

system_instruction = "Don't include names from the conversation and try to be very precise with your answers and give very similar answers to examples given in the file. Don't go on adding details about the answer, just understand the questions and answers from examples in the file, and when answering for question in prompt use your understanding from examples from the file to answer the prompt question accurately like answers in examples and  follow this main prompt for every question- \"Detect where the context changed in the sentence and return that short part of the conversation like examples in file. If you don't find any change in the context then return \"No change\""

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    audio_file_path = "uploaded_audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text_result = convert_audio_to_text(audio_file_path)
    st.write("Transcription:")
    st.write(text_result)

    if api_key:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction, safety_settings=safety_settings)

        convo = model.start_chat(history=[
            {"role": "user", "parts": extract_pdf_pages("sample_data.txt")}
        ])
        convo.send_message(text_result)

        last_response = convo.last.candidates[0].content.parts[0].text
        st.write("AI Response:")
        st.write(last_response)