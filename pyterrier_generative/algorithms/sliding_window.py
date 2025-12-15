"""Sliding window ranking algorithm."""

import pandas as pd
import numpy as np

from pyterrier_generative.algorithms.common import RankedList, iter_windows


def sliding_window(model, query: str, query_results: pd.DataFrame):
    """
    Sliding window algorithm for ranking documents.
    Note: This is only used when batching is disabled.
    When batching is enabled, collect_windows_for_batching handles window collection.
    """
    qid = query_results['qid'].iloc[0]
    query_results = query_results.sort_values('score', ascending=False)
    doc_idx = query_results['docno'].to_numpy()
    doc_texts = query_results['text'].to_numpy()
    ranking = RankedList(doc_idx, doc_texts)

    # Process each window sequentially
    for start_idx, end_idx, window_len in iter_windows(len(query_results), model.window_size, model.stride):
        kwargs = {
            'qid': qid,
            'query': query,
            'doc_text': ranking[start_idx:end_idx].doc_texts.tolist(),
            'doc_idx': ranking[start_idx:end_idx].doc_idx.tolist(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_len': window_len
        }
        order = np.array(model(**kwargs))
        new_idxs = start_idx + order
        orig_idxs = np.arange(start_idx, end_idx)
        ranking[orig_idxs] = ranking[new_idxs]

    return ranking.doc_idx, ranking.doc_texts


__all__ = ['sliding_window']
