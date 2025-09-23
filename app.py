from transformers import pipeline


def fetchTextLabelUsingBert(text: str, labels=list[str]):
    model_id = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_id)
    result = classifier(text, candidate_labels=labels)

    return result["labels"][0]
