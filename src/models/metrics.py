from rouge import Rouge

def score_summary(summary, reference):
    """
    Computes ROUGE scores for the generated summary against the reference.
    """
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores[0]
