# Reconnaissance-Automatique-de-la-Parole-et-Analyse-de-Sentiment
Ce projet a pour objectif de convertir des enregistrements audio en texte à l'aide d'un modèle de reconnaissance automatique de la parole (ASR) et de prédire le sentiment exprimé dans le texte transcrit.

# Modèles Utilisés
Modèle de Reconnaissance Automatique de la Parole (ASR)

**Nom du modèle* : wav2vec2-large-xlsr-53-french
**Description*: Ce modèle, fourni par huggingsound, est utilisé pour la transcription de fichiers audio en texte. Il est spécialement conçu pour les langues françaises.

Modèle d'Analyse de Sentiment

**Nom du modèle* : distilbert-base-uncased
**Description* : Ce modèle est une version allégée de BERT (DistilBERT) utilisée pour la classification des **sentiments*: Le modèle a été fine-tuné sur un ensemble de données spécifique pour cette tâche.
**Performance* :
Accuracy : 0,94
Loss : 0,16


# Fonctionnalités
Reconnaissance vocale (ASR) : Utilisation du modèle wav2vec2-large-xlsr-53-french pour convertir des fichiers audio en texte.
Analyse de sentiment : Utilisation d'un modèle DistilBERT pour classifier le sentiment des textes transcrits en positif ou négatif.

# Prérequis

Python 3.7 ou version ultérieure
Pytorch
Transformers (Hugging Face)
huggingsound
Installation
Clonez le dépôt :

```
git clone https://github.com/LaurianeMD/Reconnaissance-Automatique-de-la-Parole-et-Analyse-de-Sentiment.git
cd Reconnaissance-Automatique-de-la-Parole-et-Analyse-de-Sentiment
```

# Installez les dépendances nécessaires :

```
pip install -r requirements.txt
```
# Modèle pré-entraîné
Téléchargez le modèle pré-entraîné DistilBERT pour l'analyse de sentiment à partir de ce lien: https://drive.google.com/file/d/1p0Mxrof1xCehG7C3AQbKrR_qLBzOtk2N/view?usp=sharing
Ensuite, placez le fichier téléchargé dans le répertoire racine du projet.

# Utilisation
Convertir l'audio en texte et prédire le sentiment :

Exécutez le script predict_from_audio.py pour transcrire l'audio et prédire le sentiment :

```
python predict_from_audio.py
Modifier les fichiers audio :
```
Vous pouvez remplacer le fichier audio_1.wav par n'importe quel autre fichier audio que vous souhaitez analyser. Assurez-vous que le chemin du fichier est correctement défini dans le script predict_from_audio.py.

# Structure du Projet
**predict_from_audio.py** : Script principal pour la transcription et la prédiction de sentiment.
**model_sentiment.py** : Script d'entraînement du modèle de classification de sentiment à partir de texte.
**save_model.py** : Script pour sauvegarder le modèle entraîné.
**requirements.txt** : Liste des dépendances nécessaires pour exécuter le projet.
**model_sentiment.py**: Script d'entraînement du modèle de classification de sentiment à partir du dataset https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews .

# Auteurs
Lauriane MBAGDJE DORENAN

# Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus d'informations.