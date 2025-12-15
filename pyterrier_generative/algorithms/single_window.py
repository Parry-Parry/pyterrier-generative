"""Single window ranking algorithm."""

import pandas as pd
import numpy as np
from numpy import concatenate as concat


def single_window(model, query: str, query_results: pd.DataFrame):
    """
    Single window ranking algorithm.
    Ranks the top window_size documents and leaves the rest unchanged.
    """
    qid = query_results['qid'].iloc[0]
    query_results = query_results.sort_values('score', ascending=False)
    candidates = query_results.iloc[:model.window_size]
    rest = query_results.iloc[model.window_size:]
    doc_idx = candidates['docno'].to_numpy()
    doc_texts = candidates['text'].to_numpy()
    rest_idx = rest['docno'].to_numpy()
    rest_texts = rest['text'].to_numpy()

    kwargs = {
        'qid': qid,
        'query': query,
        'doc_text': doc_texts.tolist(),
        'doc_idx': doc_idx.tolist(),
        'start_idx': 0,
        'end_idx': len(doc_texts),
        'window_len': len(doc_texts)
    }
    order = np.array(model(**kwargs))
    orig_idxs = np.arange(0, len(doc_texts))
    doc_idx[orig_idxs] = doc_idx[order]
    doc_texts[orig_idxs] = doc_texts[order]

    return concat([doc_idx, rest_idx]), concat([doc_texts, rest_texts])


__all__ = ['single_window']
