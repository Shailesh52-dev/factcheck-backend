import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Load and Merge Data
try:
    print("Loading datasets...")
    # Load the two separate files
    real_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

    # Add labels: 0 for Real, 1 for Fake
    real_df['label'] = 0
    fake_df['label'] = 1

    # Combine them
    df = pd.concat([real_df, fake_df], ignore_index=True)

    # Shuffle the dataset (Important! otherwise model learns order)
    df = df.sample(frac=1).reset_index(drop=True)

    # We generally rely on the 'text' column, but sometimes 'title' is useful too.
    # For this guide, we use 'text'.
    df = df[['text', 'label']]
    
    print(f"Data loaded. Total articles: {len(df)}")
    
except FileNotFoundError:
    print("Error: Could not find 'True.csv' or 'Fake.csv'. Make sure they are in this folder.")
    exit()

# Split the data
# We use a small subset for testing if you want to be quick, or all data for accuracy.
# To speed up training on a laptop, let's limit to 10,000 samples for now (Optional)
# df = df.head(10000) 

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# 2. Tokenization
print("Tokenizing data (this may take a moment)...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# 3. Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # 3 epochs is standard
    per_device_train_batch_size=8,   # Low batch size for GTX 1650 (4GB VRAM)
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # fp16=True,                     # Uncomment this line if you have a GPU (Speeds up training significantly)
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")
trainer.train()

# 4. Save the Model
print("Saving model to ./fake_news_model...")
model.save_pretrained("./fake_news_model")
tokenizer.save_pretrained("./fake_news_model")
print("Done! You can now run main.py")