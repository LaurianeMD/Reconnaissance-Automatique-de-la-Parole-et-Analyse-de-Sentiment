import streamlit as st
from transformers import pipeline
from huggingsound import SpeechRecognitionModel

# Charger le modèle de reconnaissance vocale
asr_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Charger le pipeline de sentiment
sentiment_pipeline = pipeline("sentiment-analysis", model="VOTRE-NOM-UTILISATEUR/distilbert-sentiment-model")

st.title("Transcription et Analyse de Sentiment")

# Télécharger l'audio
audio_file = st.file_uploader("Téléchargez votre fichier audio", type=["wav", "mp3"])

if audio_file:
    # Transcrire l'audio
    transcriptions = asr_model.transcribe([audio_file.name])
    transcription = transcriptions[0]['transcription']
    st.write("Transcription : ", transcription)

    # Analyser le sentiment
    sentiment = sentiment_pipeline(transcription)
    st.write("Sentiment : ", sentiment[0])
