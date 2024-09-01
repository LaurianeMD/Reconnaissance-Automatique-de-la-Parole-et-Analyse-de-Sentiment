import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from tqdm import tqdm
import logging

# Charger les données
train_df = pd.read_csv("Dataset/train.csv")  # Assurez-vous que les chemins de fichiers sont corrects
val_df = pd.read_csv("Dataset/valid.csv")
test_df = pd.read_csv("Dataset/test.csv")

# Utiliser le tokenizer de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Créer un Dataset personnalisé
class CustomDataset(Dataset):
    def __init__(self, df):
        self.texts = df['review'].tolist()
        self.labels = df['polarity'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        return encoding, torch.tensor(label)

# Créer les datasets
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)
test_dataset = CustomDataset(test_df)

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Configurer le niveau de journalisation
logging.basicConfig(level=logging.INFO)  # Changez à DEBUG pour plus de détails

# Initialisation des composants avec DistilBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Fonction d'entraînement et d'évaluation restent les mêmes
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        # Extraire les données correctement
        inputs = batch[0]
        labels = batch[1]

        input_ids = inputs['input_ids'].squeeze(1).to(device)
        attention_mask = inputs['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        # Exécuter la passe avant (forward pass)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        # Backpropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch[0]
            labels = batch[1]

            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            # Exécuter la passe avant (forward pass)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Entraînement et évaluation
num_epochs = 3
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Évaluation finale sur les données de test
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), 'distilbert_sentiment_model.pth')
