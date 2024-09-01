import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingsound import SpeechRecognitionModel

# --- 1. Reconnaissance Vocale (ASR) ---
# Initialiser le modèle de reconnaissance vocale
asr_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Chemin vers le fichier audio
audio_paths = ["audio_1.wav"]

# Transcrire l'audio en texte
try:
    transcriptions = asr_model.transcribe(audio_paths)
    print("Transcriptions:", transcriptions)
except Exception as e:
    print("Erreur lors de la transcription:", str(e))

# Extraire le texte transcrit
transcribed_texts = [transcription['transcription'] for transcription in transcriptions]
print(f'Texte transcrit: {transcribed_texts}')

# --- 2. Chargement du Modèle Enregistré ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le tokenizer et le modèle pré-entraîné
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Charger les poids du modèle sauvegardé
model.load_state_dict(torch.load('distilbert_sentiment_model.pth', map_location=device))
model.to(device)
model.eval()

# --- 3. Prédiction du Sentiment sur le Texte Transcrit ---
for text in transcribed_texts:
    encoding = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to(device)
    
    with torch.no_grad():
        outputs = model(**encoding)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    # Interpréter le résultat de la prédiction
    sentiment = "positif" if prediction == 1 else "négatif"
    print(f'Texte: "{text}" | Sentiment prédit: {sentiment}')
