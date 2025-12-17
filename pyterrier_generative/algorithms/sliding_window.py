"""Sliding window ranking algorithm."""

import pandas as pd
import numpy as np

from pyterrier_generative.algorithms.common import RankedList, iter_windows


def sliding_window(model, query: str, query_results: pd.DataFrame):
    """
    Sliding window algorithm for ranking documents.
    Note: This is only used when batching is disabled.
    When batching is enabled, the batched version handles window collection.
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


def initialize_sliding_window_queries(model, inp: pd.DataFrame):
    """
    Initialize query states for batched sliding window processing.

    Args:
        model: The generative ranker model
        inp: Input DataFrame with all queries

    Returns:
        Dict mapping qid -> query state dict
    """
    queries_state = {}

    for qid, query_group in inp.groupby('qid'):
        query = query_group['query'].iloc[0]
        query_results = query_group.sort_values('score', ascending=False)

        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_idx, doc_texts)

        # Precompute all window positions for this query
        window_positions = list(iter_windows(len(query_results), model.window_size, model.stride))

        queries_state[qid] = {
            'qid': qid,
            'query': query,
            'ranking': ranking,
            'window_positions': window_positions,
            'current_window_idx': 0,
            'total_windows': len(window_positions)
        }

    return queries_state


def collect_sliding_window_batch(queries_state, window_idx):
    """
    Collect the window at position window_idx from all queries.

    Args:
        queries_state: Dict mapping qid -> query state
        window_idx: Which window to collect (0 = first window, 1 = second, etc.)

    Returns:
        List of window dicts for batching
    """
    windows = []

    for qid, state in queries_state.items():
        # Check if this query has a window at this index
        if window_idx < state['total_windows']:
            start_idx, end_idx, window_len = state['window_positions'][window_idx]

            # Extract window content from CURRENT ranking state
            ranking = state['ranking']
            window_docs = ranking[start_idx:end_idx]

            kwargs = {
                'qid': qid,
                'query': state['query'],
                'doc_text': window_docs.doc_texts.tolist(),
                'doc_idx': window_docs.doc_idx.tolist(),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'window_len': window_len
            }

            windows.append({
                'qid': qid,
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'kwargs': kwargs
            })

    return windows


def apply_sliding_window_batch_results(windows_data, orders, queries_state):
    """
    Apply batched ranking results to query states.

    Args:
        windows_data: List of window dicts from collect function
        orders: List of ranking orders from model
        queries_state: Dict to update in-place
    """
    for window_data, order in zip(windows_data, orders):
        qid = window_data['qid']
        state = queries_state[qid]
        ranking = state['ranking']

        start_idx = window_data['start_idx']
        end_idx = window_data['end_idx']
        order = np.array(order)

        # Apply reordering within the window
        new_idxs = start_idx + order
        orig_idxs = np.arange(start_idx, end_idx)
        ranking[orig_idxs] = ranking[new_idxs]


def sliding_window_batched_iteration(model, queries_state):
    """
    Execute sliding window processing across all queries with cross-query batching.

    This processes one window position at a time across all queries, ensuring that
    windows from the same query are never batched together (as they depend on each other).

    Args:
        model: The generative ranker model
        queries_state: Dict mapping qid -> query state
    """
    # Find the maximum number of windows across all queries
    max_windows = max(state['total_windows'] for state in queries_state.values())

    # Process each window position across all queries
    for window_idx in range(max_windows):
        # Collect window at position window_idx from all queries that have it
        windows = collect_sliding_window_batch(queries_state, window_idx)

        if not windows:
            break

        # Batch rank all windows at this position
        kwargs_list = [w['kwargs'] for w in windows]
        orders = model._rank_windows_batch(kwargs_list)

        # Apply results back to query states
        apply_sliding_window_batch_results(windows, orders, queries_state)


def convert_sliding_window_states_to_windows(queries_state):
    """
    Convert query states to windows_data format for apply_batched_results.

    Args:
        queries_state: Dict mapping qid -> query state

    Returns:
        List of window dicts compatible with apply_batched_results
    """
    all_windows = []

    for qid, state in queries_state.items():
        ranking = state['ranking']

        all_windows.append({
            'qid': qid,
            'query': state['query'],
            'ranking': ranking,
            'window_info': (0, len(ranking)),
            'kwargs': {},
            'sliding_window_state': True  # Marker for sliding window results
        })

    return all_windows


__all__ = [
    'sliding_window',
    'initialize_sliding_window_queries',
    'sliding_window_batched_iteration',
    'convert_sliding_window_states_to_windows'
]
