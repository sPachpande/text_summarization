from transformers import pipeline
import torch

def summarize_text(content, model_name, device=0):
    """
    Summarizes the given content using the specified model.
    """
    summarizer = pipeline("summarization", model=model_name, device=device)
    return summarizer(content, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
