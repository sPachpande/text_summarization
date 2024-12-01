import matplotlib.pyplot as plt

def plot_rouge_scores(models, scores, output_path):
    """
    Plots and saves bar graphs for ROUGE scores.
    """
    plt.figure(figsize=(15, 5))
    metrics = ['rouge-1', 'rouge-2', 'rouge-l']

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        metric_scores = [scores[model][metric] for model in models]
        plt.bar(models, metric_scores, color=['blue', 'orange', 'green'])
        plt.title(f"Average {metric.upper()} Scores")
        plt.ylabel("Score")
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
