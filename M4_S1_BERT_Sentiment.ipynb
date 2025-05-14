# Step 1: Install libraries (Colab-compatible)
!pip install transformers datasets torch

# Step 2: Load the IMDB dataset
from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
print("Sample review:", imdb_dataset['train'][0]['text'][:100] + "...")
print("Sample label:", imdb_dataset['train'][0]['label'])

# Step 3: Load pre-trained BERT model and tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set descriptive label names
model.config.id2label = {0: "negative", 1: "positive"}
model.config.label2id = {"negative": 0, "positive": 1}

# Step 4: Preprocess the dataset
def preprocess_reviews(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_imdb = imdb_dataset.map(preprocess_reviews, batched=True)

# Step 5: Quick inference test with GPU support
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sample_review = "This movie was absolutely fantastic! The acting was superb and the plot was engaging."
inputs = tokenizer(sample_review, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_id = torch.argmax(logits, dim=1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(f"Review: '{sample_review}'")
print(f"Predicted sentiment (ID): {predicted_class_id}")
print(f"Predicted sentiment (Label): {predicted_label}")
