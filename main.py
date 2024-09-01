from save_model import transcribe_audio
from sentiment_analysis import analyze_sentiment

if __name__ == "__main__":
    # Chemin vers le fichier audio à transcrire
    file_path = "path_to_audio_file.wav"
    
    # Transcrire l'audio
    transcription = transcribe_audio(file_path)
    print("Transcription:", transcription)
    
    # Analyser le sentiment de la transcription
    sentiment_result = analyze_sentiment(transcription)
    print("Sentiment:", sentiment_result)
