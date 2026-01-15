from src.evaluation.metrics import recall, precision, citation_accuracy

def test_recall():
    assert recall([1,2,3], [2]) == 1.0
    assert recall([1,2,3], [5]) == 0.0

def test_precision():
    assert precision([1,2,3], [2,3]) == 2/3

def test_citation_accuracy():
    assert citation_accuracy([2], [1,2,3]) == 1.0
    assert citation_accuracy([4], [1,2,3]) == 0.0