import torch
from bertviz import head_view
from transformers import AutoTokenizer, AutoModel

# Step 1: Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()  # Set model to evaluation mode for inference

# Step 2: Prepare the input sentence
sentence = "Elena wants to build a better translation app."
inputs = tokenizer.encode_plus(
    sentence,
    return_tensors="pt",
    add_special_tokens=True
)
input_ids = inputs["input_ids"]

# Step 3: Run the model to get attention outputs
with torch.no_grad():
    outputs = model(input_ids)
    attention = outputs.attentions  # Tuple of attention weights for each layer

# Step 4: Convert input IDs to tokens for visualization
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Step 5: Visualize attention weights using BertViz head_view
head_view(attention, tokens)
