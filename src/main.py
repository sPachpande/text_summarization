import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loaders.cnn_dataset import load_cnn_dailymail
from data_loaders.samsum_dataset import load_samsum
from models.summarizer import summarize_text
from models.metrics import score_summary
from visualizations.plot_results import plot_rouge_scores
import numpy as np
import torch

def evaluate_dataset(dataset, models, device=0):
    """
    Evaluates the dataset using the specified models and returns average ROUGE scores.
    """
    scores = {model: {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0} for model in models}
    counts = {model: 0 for model in models}

    for entry in dataset:
        text = entry['dialogue'] if 'dialogue' in entry else entry['article']
        reference = entry['summary'] if 'summary' in entry else entry['highlights']
        
        for model in models:
            try:
                summary = summarize_text(text, model, device)
                rouge_scores = score_summary(summary, reference)
                for metric in scores[model]:
                    scores[model][metric] += rouge_scores[metric]['f']
                counts[model] += 1
            except Exception as e:
                print(f"Error with {model}: {e}")

    # Average the scores
    for model in models:
        for metric in scores[model]:
            scores[model][metric] /= max(counts[model], 1)

    return scores

def main():
    models = ['facebook/bart-large-cnn', 'Falconsai/text_summarization', 'google/pegasus-large']
    device = 0 if torch.cuda.is_available() else -1

    print("Loading datasets...")
    cnn_dataset = load_cnn_dailymail()
    samsum_dataset = load_samsum()

    print("Evaluating CNN/DailyMail dataset...")
    cnn_scores = evaluate_dataset(cnn_dataset, models, device)
    plot_rouge_scores(models, cnn_scores, "results/cnn_dailymail_rouge_scores.png")

    print("Evaluating SAMSum dataset...")
    samsum_scores = evaluate_dataset(samsum_dataset, models, device)
    plot_rouge_scores(models, samsum_scores, "results/samsum_rouge_scores.png")

    print("Evaluation complete. Results saved in the results/ folder.")

if __name__ == "__main__":
    main()
