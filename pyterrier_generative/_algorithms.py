from enum import Enum
import logging 

import pandas as pd
import numpy as np
from numpy import concatenate as concat
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def iter_windows(n, window_size, stride, verbose : bool = False):
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride, disable=verbose), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len

def split(l, i):
    return l[:i], l[i:]

class RankedList(object):
    def __init__(self, doc_idx=None, doc_texts=None) -> None:
        self.doc_texts = doc_texts if doc_texts is not None else np.array([])
        self.doc_idx = doc_idx if doc_idx is not None else np.array([])
    
    def __len__(self):
        return len(self.doc_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RankedList(self.doc_idx[key], self.doc_texts[key])
        elif isinstance(key, int):
            return RankedList(np.array([self.doc_idx[key]]), np.array([self.doc_texts[key]]))
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            return RankedList([self.doc_idx[i] for i in key], [self.doc_texts[i] for i in key])
        else:
            raise TypeError("Invalid key type. Please use int, slice, list, or numpy array.")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.doc_idx[key], self.doc_texts[key] = value.doc_idx[0], value.doc_texts[0]
        elif isinstance(key, slice):
            self.doc_idx[key], self.doc_texts[key] = value.doc_idx, value.doc_texts
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            if len(key) != len(value):
                raise ValueError("Assigning RankedList requires the same length as the key.")
            for i, idx in enumerate(key):
                self.doc_idx[idx], self.doc_texts[idx] = value.doc_idx[i], value.doc_texts[i]

    def __add__(self, other):
        print(type(other))
        if not isinstance(other, RankedList):
            raise TypeError("Unsupported operand type(s) for +: 'RankedList' and '{}'".format(type(other)))
        return RankedList(concat([self.doc_idx, other.doc_idx]), concat([self.doc_texts, other.doc_texts]))
      
    def __str__(self):
      return f"{self.doc_idx}, {self.doc_texts}"

def sliding_window(model, query : str, query_results : pd.DataFrame):
        qid = query_results['qid'].iloc[0]
        query_results = query_results.sort_values('score', ascending=False)
        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_texts, doc_idx)

        # Check if model supports batched ranking
        if hasattr(model, '_rank_windows_batch'):
            # Collect all windows first
            windows_data = []
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
                windows_data.append((start_idx, end_idx, kwargs))

            # Batch process all windows at once
            windows_kwargs = [w[2] for w in windows_data]
            orders = model._rank_windows_batch(windows_kwargs)

            # Apply rankings
            for (start_idx, end_idx, _), order in zip(windows_data, orders):
                order = np.array(order)
                new_idxs = start_idx + order
                orig_idxs = np.arange(start_idx, end_idx)
                ranking[orig_idxs] = ranking[new_idxs]
        else:
            # Fallback to single-window processing
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
    
def single_window(model, query : str, query_results : pd.DataFrame):
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
    
# from https://github.com/ielab/llm-rankers/blob/main/llmrankers/setwise.py

def _heapify(model, query, ranking, n, i):
    # Find largest among root and children
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    li_comp = model(**{
        'query': query['query'].iloc[0],
        'doc_text': [ranking.doc_texts[i], ranking.doc_texts[l]],
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
    if l < n and li_comp == 0: largest = l
    if r < n and rl_comp == 0: largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        ranking[i], ranking[largest] = ranking[largest], ranking[i]
        model._heapify(query, ranking, n, largest)

def setwise(model, query : str, query_results : pd.DataFrame):
    query_results = query_results.sort_values('score', ascending=False)
    doc_idx = query_results['docno'].to_numpy()
    doc_texts = query_results['text'].to_numpy()
    ranking = RankedList(doc_texts, doc_idx)
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

def _tdpart_pivot_pos(model) -> int:
    """
    Returns the pivot position (0-indexed) within the initial window.

    Expected convention:
      - model.cutoff is the desired top-k cutoff (e.g., 10 => pivot_pos=9)

    If you already store cutoff as an index (cutoff-1), set:
      - model.cutoff_is_index = True
    """
    window_size = int(getattr(model, "window_size", 20))
    cutoff = int(getattr(model, "cutoff", 10))

    if getattr(model, "cutoff_is_index", False):
        pivot_pos = cutoff
    else:
        pivot_pos = cutoff - 1

    # Clamp into [0, window_size-1]
    if window_size <= 1:
        return 0
    return max(0, min(pivot_pos, window_size - 1))


def _tdpart_step(model, qid: str, query: str, ranking: "RankedList"):
    """
    One TDPart iteration on the current candidate pool.

    Returns:
      c: RankedList of current candidates (best-so-far segment)
      b: RankedList of backfill / remainder
      done: bool indicating whether cutoff is finalized
    """
    window_size = int(getattr(model, "window_size", 20))
    buffer_budget = int(getattr(model, "buffer", 20))
    pivot_pos = _tdpart_pivot_pos(model)

    # l: current window, r: remainder
    l = ranking[:window_size]
    r = ranking[window_size:]

    # Initial sort of l
    kwargs = {
        "qid": qid,
        "query": query,
        "doc_text": l.doc_texts.tolist(),
        "doc_idx": l.doc_idx.tolist(),
        "start_idx": 0,
        "end_idx": len(l),
        "window_len": len(l),
    }
    order = np.asarray(model(**kwargs))
    orig = np.arange(len(l))
    l[orig] = l[order]

    # If we never filled a full window, a single sort is enough
    if len(l) < window_size:
        return l, r, True

    # Pivot + partitions
    p = l[pivot_pos]                 # RankedList of length 1
    c = l[:pivot_pos]                # candidates better than pivot (in current view)
    b = l[pivot_pos + 1 :]           # backfill worse than pivot (in current view)

    # We re-score windows of size (window_size-1) plus the pivot => total window_size
    sub_window_size = window_size - 1

    # Grow c until we hit buffer_budget or we exhaust r
    # For batched mode, we need to re-do this sequentially since we need pivot results
    while len(c) < buffer_budget and len(r) > 0:
        l2, r = split(r, sub_window_size)   # l2 is RankedList
        l2 = p + l2                         # inject pivot into this window

        kwargs = {
            "qid": qid,
            "query": query,
            "doc_text": l2.doc_texts.tolist(),
            "doc_idx": l2.doc_idx.tolist(),
            "start_idx": 0,
            "end_idx": len(l2),
            "window_len": len(l2),
        }
        order = np.asarray(model(**kwargs))
        orig = np.arange(len(l2))
        l2[orig] = l2[order]

        # Find pivot location after sort
        # (p is length-1 RankedList; compare underlying id)
        pivot_id = p.doc_idx[0]
        p_idx = int(np.where(l2.doc_idx == pivot_id)[0][0])

        # Left of pivot beats pivot; right does not (for this window)
        c = c + l2[:p_idx]
        b = b + l2[p_idx + 1 :]

    # If we never found anything better than pivot beyond the initial c,
    # then top-(pivot_pos+1) is finalized.
    if len(c) == pivot_pos:
        top = c + p
        bottom = b + r
        return top, bottom, True

    # Otherwise, we have more candidates than budget: keep first buffer_budget and
    # push the rest (plus pivot and all known-worse) into backfill.
    c_keep, c_extra = split(c, buffer_budget)
    backfill = c_extra + p + b + r
    return c_keep, backfill, False


def tdpart(model, query: str, query_results: pd.DataFrame):
    """
    Standalone TDPart (partition rank) driver.

    Required model attrs:
      - window_size: int
      - buffer: int
      - cutoff: int (desired top-k cutoff, not index) OR set model.cutoff_is_index=True
      - max_iters: int (optional; default 100)

    model(**kwargs) must return an ordering (permutation) over the provided window.
    """
    qid = query_results["qid"].iloc[0]
    max_iters = int(getattr(model, "max_iters", 100))

    query_results = query_results.sort_values("score", ascending=False)
    doc_idx = query_results["docno"].to_numpy()
    doc_texts = query_results["text"].to_numpy()

    c = RankedList(doc_idx, doc_texts)
    b = RankedList()

    done = False
    iters = 0
    while not done and iters < max_iters:
        iters += 1
        c, b_new, done = _tdpart_step(model, qid, query, c)
        b = b + b_new

    if iters == max_iters:
        logger.warning("TDPart reached max_iters for qid=%s", qid)

    out = c + b
    return out.doc_idx, out.doc_texts

class Algorithm(Enum):
    SLIDING_WINDOW = "sliding_window"
    SINGLE_WINDOW = "single_window"
    SETWISE = "setwise"
    TDPART = "tdpart"