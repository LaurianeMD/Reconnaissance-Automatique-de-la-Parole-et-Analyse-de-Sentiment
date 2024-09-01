import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Charger le modèle de sentiment
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('distilbert_sentiment_model.pth'))

# Enregistrer le modèle dans le format Hugging Face
model.save_pretrained("distilbert-sentiment")

# Charger le tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained("distilbert-sentiment")
