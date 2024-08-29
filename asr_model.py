from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# Charger le modèle et le processeur
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

def transcribe_audio(file_path):
    # Charger un fichier audio
    speech_array, sampling_rate = torchaudio.load(file_path)
    # Prétraitement de l'audio
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    
    # Faire l'inférence
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Décoder la transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

if __name__ == "__main__":
    file_path = "path_to_audio_file.wav"
    transcription = transcribe_audio(file_path)
    print("Transcription:", transcription)
