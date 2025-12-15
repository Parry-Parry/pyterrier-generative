"""Setwise ranking algorithm using heapsort."""

import pandas as pd

from pyterrier_generative.algorithms.common import RankedList


def _heapify(model, query, ranking, n, i):
    """Heapify helper function for setwise algorithm."""
    # Find largest among root and children
    largest = i
    left = 2 * i + 1
    r = 2 * i + 2
    li_comp = model(**{
        'query': query['query'].iloc[0],
        'doc_text': [ranking.doc_texts[i], ranking.doc_texts[left]],
        'start_idx': 0,
        'end_idx': 1,
        'window_len': 2
    })
    rl_comp = model(**{
        'query': query['query'].iloc[0],
        'doc_text': [ranking.doc_texts[r], ranking.doc_texts[largest]],
        'start_idx': 0,
        'end_idx': 1,
        'window_len': 2
    })
    if left < n and li_comp == 0:
        largest = left
    if r < n and rl_comp == 0:
        largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        ranking[i], ranking[largest] = ranking[largest], ranking[i]
        model._heapify(query, ranking, n, largest)


def setwise(model, query: str, query_results: pd.DataFrame):
    """
    Setwise ranking algorithm using heapsort.
    From https://github.com/ielab/llm-rankers/blob/main/llmrankers/setwise.py
    """
    query_results = query_results.sort_values('score', ascending=False)
    doc_idx = query_results['docno'].to_numpy()
    doc_texts = query_results['text'].to_numpy()
    ranking = RankedList(doc_idx, doc_texts)
    n = len(query_results)
    ranked = 0
    # Build max heap
    for i in range(n // 2, -1, -1):
        _heapify(model, query, ranking, n, i)
    for i in range(n - 1, 0, -1):
        # Swap
        ranking[i], ranking[0] = ranking[0], ranking[i]
        ranked += 1
        if ranked == model.k:
            break
        # Heapify root element
        _heapify(model, query, ranking, i, 0)
    return ranking.doc_idx, ranking.doc_texts


__all__ = ['setwise']
