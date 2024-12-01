from datasets import load_dataset

def load_samsum():
    """Loads the SAMSum dataset."""
    return load_dataset("samsum", split="test[:50000]")
