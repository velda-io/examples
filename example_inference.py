#!/usr/bin/env python 

# This is a simple training script with PyTorch for a CNN model
# To train with GPU, run:
#    vrun -P gpu-t4-1 ./example_inference.py
# 
import torch
from transformers import pipeline

# 1. Check for GPU availability and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device_id = 0 # Use GPU 0
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")
    device_id = -1 # Use CPU

print(f"Inference will run on: {'GPU' if device_id != -1 else 'CPU'}")

# 2. Load a pre-trained sentiment analysis model using Hugging Face pipeline
# The 'pipeline' abstraction makes inference incredibly easy.
# We'll use 'distilbert-base-uncased-finetuned-sst-2-english', which is small and efficient.
print(f"\nLoading model 'distilbert-base-uncased-finetuned-sst-2-english'...")
# The device argument ensures the model is loaded directly onto the GPU if available
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device_id)

print("Model loaded successfully.")

# 3. Prepare input text for inference
texts_to_analyze = [
    "I love Velda, it's absolutely fantastic!",
    "This movie was really boring and a waste of time.",
    "The weather today is pretty standard, nothing special.",
    "I am so incredibly happy with the results of this experiment."
]

print("\nPerforming inference on the following texts:")
for text in texts_to_analyze:
    print(f"- \"{text}\"")

# 4. Perform inference
# The pipeline handles tokenization, running the model, and post-processing.
results = sentiment_pipeline(texts_to_analyze)

# 5. Print results
print("\nInference Results:")
for i, result in enumerate(results):
    print(f"Text: \"{texts_to_analyze[i]}\"")
    print(f"  Sentiment: {result['label']} (Score: {result['score']:.4f})")
    print("-" * 30)

print("\nInference complete!")