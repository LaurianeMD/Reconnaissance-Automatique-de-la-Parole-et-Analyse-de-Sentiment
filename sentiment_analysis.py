from transformers import pipeline

# Charger le pipeline pour l'analyse de sentiment
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    # Analyser le sentiment du texte
    sentiment = sentiment_pipeline(text)
    return sentiment

if __name__ == "__main__":
    # Exemple de texte pour l'analyse de sentiment
    example_text = "Exemple de transcription à analyser."
    sentiment_result = analyze_sentiment(example_text)
    print("Sentiment:", sentiment_result)
