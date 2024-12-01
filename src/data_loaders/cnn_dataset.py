from datasets import load_dataset

def load_cnn_dailymail():
    """Loads the CNN/DailyMail dataset."""
    return load_dataset("cnn_dailymail", "3.0.0", split="test[:50000]")
