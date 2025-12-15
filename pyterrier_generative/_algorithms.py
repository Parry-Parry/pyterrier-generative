"""Algorithm coordination and cross-query batching infrastructure."""

from enum import Enum
import pandas as pd
import numpy as np

# Import from algorithms subpackage
from pyterrier_generative.algorithms.common import RankedList, iter_windows
from pyterrier_generative.algorithms.sliding_window import sliding_window
from pyterrier_generative.algorithms.single_window import single_window
from pyterrier_generative.algorithms.setwise import setwise
from pyterrier_generative.algorithms.tdpart import (
    tdpart,
    initialize_tdpart_queries,
    tdpart_batched_iteration,
    tdpart_final_collation_batched,
    convert_tdpart_states_to_windows,
)


class Algorithm(Enum):
    SLIDING_WINDOW = "sliding_window"
    SINGLE_WINDOW = "single_window"
    SETWISE = "setwise"
    TDPART = "tdpart"


def collect_windows_for_batching(model, inp: pd.DataFrame):
    """
    Collect all ranking windows from all queries for cross-query batching.

    Args:
        model: The generative ranker model
        inp: Input DataFrame with all queries

    Returns:
        List of dicts, each containing:
            - 'qid': query ID
            - 'query': query text
            - 'ranking': RankedList for this query
            - 'window_info': (start_idx, end_idx) for this window
            - 'kwargs': keyword arguments for ranking this window
    """
    from pyterrier_generative.modelling.base import Algorithm as AlgorithmImport

    all_windows = []

    for qid, query_group in inp.groupby('qid'):
        query = query_group['query'].iloc[0]
        query_results = query_group.sort_values('score', ascending=False)

        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_idx, doc_texts)

        # Collect windows based on algorithm
        if model.algorithm == AlgorithmImport.SLIDING_WINDOW:
            # Multiple windows per query
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
                all_windows.append({
                    'qid': qid,
                    'query': query,
                    'ranking': ranking,
                    'window_info': (start_idx, end_idx),
                    'kwargs': kwargs
                })
        elif model.algorithm == AlgorithmImport.SINGLE_WINDOW:
            # Single window per query
            candidates = query_results.iloc[:model.window_size]
            doc_idx = candidates['docno'].to_numpy()
            doc_texts = candidates['text'].to_numpy()

            kwargs = {
                'qid': qid,
                'query': query,
                'doc_text': doc_texts.tolist(),
                'doc_idx': doc_idx.tolist(),
                'start_idx': 0,
                'end_idx': len(doc_texts),
                'window_len': len(doc_texts)
            }
            all_windows.append({
                'qid': qid,
                'query': query,
                'ranking': ranking,
                'window_info': (0, len(doc_texts)),
                'kwargs': kwargs,
                'rest_idx': query_results.iloc[model.window_size:]['docno'].to_numpy(),
                'rest_texts': query_results.iloc[model.window_size:]['text'].to_numpy()
            })

    # Check if TDPart algorithm - handle separately
    if model.algorithm == AlgorithmImport.TDPART:
        # Initialize query states
        queries_state = initialize_tdpart_queries(model, inp)

        # Run batched iterations
        max_iters = int(getattr(model, "max_iters", 100))
        for iteration in range(max_iters):
            active = tdpart_batched_iteration(model, queries_state, iteration)
            if active == 0:
                break

        # Final collation: batch rank all candidates to get definitive top-k
        tdpart_final_collation_batched(model, queries_state)

        # Convert states to windows_data format for compatibility
        all_windows = convert_tdpart_states_to_windows(queries_state)

    return all_windows


def apply_batched_results(all_windows_data, orders):
    """
    Apply batched ranking results back to queries and build output DataFrames.

    Args:
        all_windows_data: List of window data dicts from collect_windows_for_batching
        orders: List of ranking orders corresponding to each window

    Returns:
        List of DataFrames, one per query
    """
    # Check if this is TDPart (different structure)
    if all_windows_data and 'tdpart_state' in all_windows_data[0]:
        # TDPart: rankings are already finalized in the window data
        results = []
        for window_data in all_windows_data:
            qid = window_data['qid']
            ranking = window_data['ranking']
            num_docs = len(ranking.doc_idx)
            query_results = pd.DataFrame({
                'qid': [qid] * num_docs,
                'query': [window_data['query']] * num_docs,
                'docno': ranking.doc_idx,
                'text': ranking.doc_texts,
                'rank': range(num_docs),
                'score': [num_docs - i for i in range(num_docs)]
            })
            results.append(query_results)
        return results

    # Group windows by query
    query_windows = {}
    for window_data, order in zip(all_windows_data, orders):
        qid = window_data['qid']
        if qid not in query_windows:
            query_windows[qid] = {
                'query': window_data['query'],
                'ranking': window_data['ranking'],
                'windows': []
            }
        query_windows[qid]['windows'].append((window_data, order))

    # Apply results for each query
    results = []
    for qid, query_info in query_windows.items():
        ranking = query_info['ranking']

        # Apply each window's ranking
        for window_data, order in query_info['windows']:
            start_idx, end_idx = window_data['window_info']
            order = np.array(order)

            if 'rest_idx' in window_data:
                # SINGLE_WINDOW: rerank just the top window, keep rest
                new_idxs = order
                orig_idxs = np.arange(0, len(order))
                ranking[orig_idxs] = ranking[new_idxs]
            else:
                # SLIDING_WINDOW: rerank within window
                new_idxs = start_idx + order
                orig_idxs = np.arange(start_idx, end_idx)
                ranking[orig_idxs] = ranking[new_idxs]

        # Build output DataFrame
        num_docs = len(ranking.doc_idx)
        query_results = pd.DataFrame({
            'qid': [qid] * num_docs,
            'query': [query_info['query']] * num_docs,
            'docno': ranking.doc_idx,
            'text': ranking.doc_texts,
            'rank': range(num_docs),
            'score': [num_docs - i for i in range(num_docs)]
        })
        results.append(query_results)

    return results


__all__ = [
    'Algorithm',
    'RankedList',
    'iter_windows',
    'sliding_window',
    'single_window',
    'setwise',
    'tdpart',
    'collect_windows_for_batching',
    'apply_batched_results',
]
