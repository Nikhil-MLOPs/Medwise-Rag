import re
import numpy as np

def recall(retrieved_pages, expected_pages):
    return float(any(p in retrieved_pages for p in expected_pages))

def precision(retrieved_pages, expected_pages):
    if not retrieved_pages:
        return 0.0
    relevant = sum(1 for p in retrieved_pages if p in expected_pages)
    return relevant / len(retrieved_pages)

def extract_cited_pages(text):
    return list(map(int, re.findall(r"Page\s+(\d+)", text)))

def citation_accuracy(cited_pages, retrieved_pages):
    if not cited_pages:
        return 0.0
    return float(all(p in retrieved_pages for p in cited_pages))

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0