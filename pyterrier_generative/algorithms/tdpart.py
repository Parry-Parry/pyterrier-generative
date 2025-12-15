"""TDPart (top-down partitioning) ranking algorithm with batching support."""

import logging
import pandas as pd
import numpy as np

from pyterrier_generative.algorithms.common import RankedList, split

logger = logging.getLogger(__name__)


def initialize_tdpart_queries(model, inp: pd.DataFrame):
    """
    Initialize query states for batched TDPart processing.

    Args:
        model: The generative ranker model
        inp: Input DataFrame with all queries

    Returns:
        Dict mapping qid -> query state dict
    """
    queries_state = {}
    window_size = int(getattr(model, "window_size", 20))
    buffer_budget = int(getattr(model, "buffer", 20))
    cutoff = int(getattr(model, "cutoff", 10))
    pivot_pos = _tdpart_pivot_pos(model)

    for qid, query_group in inp.groupby('qid'):
        query = query_group['query'].iloc[0]
        query_results = query_group.sort_values('score', ascending=False)

        doc_idx = query_results['docno'].to_numpy()
        doc_texts = query_results['text'].to_numpy()
        ranking = RankedList(doc_idx, doc_texts)

        queries_state[qid] = {
            'qid': qid,
            'query': query,
            'ranking': ranking,
            'window_size': window_size,
            'buffer': buffer_budget,
            'cutoff': cutoff,
            'pivot_pos': pivot_pos,
            'phase': 'initial',
            'iteration': 0,
            'candidates': RankedList(),
            'pivot': None,
            'backfill': RankedList(),
            'backfill_accumulator': RankedList(),  # Accumulates backfill across iterations
            'remainder': ranking,
            'next_window_idx': 0
        }

    return queries_state


def build_model_kwargs(qid, state, window):
    """
    Build kwargs dict for model call from query state and window.

    Args:
        qid: Query ID
        state: Query state dict
        window: RankedList window to rank

    Returns:
        Dict with model call arguments
    """
    return {
        'qid': qid,
        'query': state['query'],
        'doc_text': window.doc_texts.tolist(),
        'doc_idx': window.doc_idx.tolist(),
        'start_idx': 0,
        'end_idx': len(window),
        'window_len': len(window)
    }


def take_next_subwindow(state):
    """
    Extract next (window_size - 1) docs from state['remainder'].
    Updates state['remainder'] in place.

    Args:
        state: Query state dict

    Returns:
        RankedList of next sub-window
    """
    sub_window_size = state['window_size'] - 1
    remainder = state['remainder']

    if len(remainder) == 0:
        return RankedList()

    # Take up to sub_window_size docs
    take = min(sub_window_size, len(remainder))
    sub_window = remainder[:take]
    state['remainder'] = remainder[take:]

    return sub_window


def apply_order(window, order):
    """
    Apply ranking order to a RankedList window.

    Args:
        window: RankedList to reorder
        order: Array of indices representing new order

    Returns:
        Reordered RankedList
    """
    order = np.asarray(order)
    orig = np.arange(len(window))
    result = RankedList(window.doc_idx.copy(), window.doc_texts.copy())
    result[orig] = result[order]
    return result


def find_pivot_in_window(sorted_window, pivot_id):
    """
    Find index of pivot document in sorted window.

    Args:
        sorted_window: RankedList that has been sorted
        pivot_id: Document ID of the pivot

    Returns:
        Integer index of pivot position
    """
    pivot_idx = int(np.where(sorted_window.doc_idx == pivot_id)[0][0])
    return pivot_idx


def check_and_update_query_state(state, iteration):
    """
    Check if query is done or needs budget trimming.
    Updates state['phase'] accordingly.

    Args:
        state: Query state dict
        iteration: Current iteration number
    """
    candidates = state['candidates']
    pivot_pos = state['pivot_pos']
    buffer = state['buffer']

    # Check if cutoff stabilized
    if len(candidates) == pivot_pos:
        # Top-(pivot_pos+1) is finalized
        if state['pivot'] is not None:
            state['candidates'] = candidates + state['pivot']
        # Accumulate iteration backfill into accumulator
        iteration_backfill = state['backfill'] + state['remainder']
        state['backfill_accumulator'] = state['backfill_accumulator'] + iteration_backfill
        # Clear temporary backfill and remainder
        state['backfill'] = RankedList()
        state['remainder'] = RankedList()
        state['pivot'] = None
        state['phase'] = 'done'
        return

    # Check if over budget
    if len(candidates) > buffer:
        # Need to trim and continue next iteration
        c_keep = candidates[:buffer]
        c_extra = candidates[buffer:]

        # Accumulate backfill across iterations (like b = b + b_new in non-batched)
        iteration_backfill = c_extra
        if state['pivot'] is not None:
            iteration_backfill = iteration_backfill + state['pivot']
        iteration_backfill = iteration_backfill + state['backfill']
        iteration_backfill = iteration_backfill + state['remainder']

        state['backfill_accumulator'] = state['backfill_accumulator'] + iteration_backfill

        # For next iteration, process c_keep (like passing c to next _tdpart_step)
        state['candidates'] = c_keep
        state['ranking'] = c_keep
        state['backfill'] = RankedList()
        state['remainder'] = RankedList()  # Will be set from ranking after pivot phase
        state['pivot'] = None
        state['phase'] = 'initial'  # Start new iteration
        state['iteration'] = iteration + 1
        state['next_window_idx'] = 0
        return

    # Still growing or exhausted
    if len(state['remainder']) == 0:
        # Exhausted all documents, mark as done
        # Assemble final ranking: keep candidates as is, accumulate remaining backfill
        if state['pivot'] is not None:
            state['candidates'] = state['candidates'] + state['pivot']
            state['pivot'] = None
        # Accumulate any remaining backfill
        state['backfill_accumulator'] = state['backfill_accumulator'] + state['backfill']
        state['backfill'] = RankedList()
        state['phase'] = 'done'


def finalize_query_iteration(state):
    """
    Mark query for budget trimming after current batch completes.

    Args:
        state: Query state dict
    """
    state['budget_exceeded'] = True


def convert_tdpart_states_to_windows(queries_state):
    """
    Convert query states to windows_data format for apply_batched_results.

    Args:
        queries_state: Dict mapping qid -> query state

    Returns:
        List of window dicts compatible with apply_batched_results
    """
    all_windows = []

    for qid, state in queries_state.items():
        # Assemble final ranking: candidates + backfill_accumulator + any remaining
        final_ranking = state['candidates']
        if state['pivot'] is not None:
            final_ranking = final_ranking + state['pivot']
        final_ranking = final_ranking + state['backfill_accumulator']
        final_ranking = final_ranking + state['backfill']
        final_ranking = final_ranking + state['remainder']

        all_windows.append({
            'qid': qid,
            'query': state['query'],
            'ranking': final_ranking,
            'window_info': (0, len(final_ranking)),
            'kwargs': {},
            'tdpart_state': True  # Marker for TDPart results
        })

    return all_windows


def collect_tdpart_windows_for_batching(queries_state, phase='pivot'):
    """
    Collect ranking windows from multiple queries at different phases.

    Args:
        queries_state: Dict mapping qid -> query state
        phase: 'pivot' (initial pivot finding) or 'grow' (candidate growth)

    Returns:
        List of window dicts for batching
    """
    windows = []

    if phase == 'pivot':
        # Initial pivot finding - one window per query
        for qid, state in queries_state.items():
            if state['phase'] == 'initial':
                window = state['ranking'][:state['window_size']]
                windows.append({
                    'qid': qid,
                    'window': window,
                    'kwargs': build_model_kwargs(qid, state, window),
                    'type': 'pivot'
                })

    elif phase == 'grow':
        # Candidate growth - multiple windows, fill batch capacity
        # Maximize parallelism by collecting as many windows as possible
        for qid, state in queries_state.items():
            if state['phase'] == 'growing':
                # Can this query produce more windows?
                # Check: not over budget, has remaining docs
                while (len(state['candidates']) < state['buffer'] and
                       len(state['remainder']) > 0):

                    # Check if we would exceed budget with this window
                    # We don't know how many docs will beat pivot, but we can estimate
                    # Worst case: all (window_size - 1) docs beat pivot
                    potential_candidates = len(state['candidates']) + (state['window_size'] - 1)

                    # If potential overflow, mark for attention but still collect window
                    # Don't break inference - we'll handle overflow after batch completes
                    if potential_candidates > state['buffer']:
                        state['budget_warning'] = True

                    # Take next sub-window + pivot
                    sub_window = take_next_subwindow(state)
                    if len(sub_window) == 0:
                        break

                    window = state['pivot'] + sub_window

                    windows.append({
                        'qid': qid,
                        'window': window,
                        'kwargs': build_model_kwargs(qid, state, window),
                        'type': 'grow',
                        'sub_window_idx': state['next_window_idx']
                    })
                    state['next_window_idx'] += 1

                    # If budget warning set, stop collecting more windows from this query
                    if state.get('budget_warning', False):
                        break

    return windows


def apply_tdpart_batched_results(windows_data, orders, queries_state):
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
        window = window_data['window']

        # Apply ordering
        sorted_window = apply_order(window, order)

        if window_data['type'] == 'pivot':
            # Initial pivot finding
            pivot_pos = state['pivot_pos']
            state['pivot'] = sorted_window[pivot_pos:pivot_pos+1]  # RankedList of length 1
            state['candidates'] = sorted_window[:pivot_pos]
            state['backfill'] = sorted_window[pivot_pos + 1:]
            state['remainder'] = state['ranking'][state['window_size']:]  # Rest of original docs
            state['phase'] = 'growing'

        elif window_data['type'] == 'grow':
            # Find pivot position in sorted window
            pivot_id = state['pivot'].doc_idx[0]
            pivot_idx = find_pivot_in_window(sorted_window, pivot_id)

            # Partition around pivot
            state['candidates'] = state['candidates'] + sorted_window[:pivot_idx]
            state['backfill'] = state['backfill'] + sorted_window[pivot_idx + 1:]

            # Check budget (mark for trimming, don't trim immediately)
            if len(state['candidates']) >= state['buffer']:
                finalize_query_iteration(state)


def tdpart_batched_iteration(model, queries_state, iteration):
    """
    Execute one TDPart iteration across all queries with batching.

    Args:
        model: The generative ranker model
        queries_state: Dict mapping qid -> query state
        iteration: Current iteration number

    Returns:
        Number of queries still active
    """
    # Phase 1: Pivot finding for new queries
    pivot_windows = collect_tdpart_windows_for_batching(queries_state, 'pivot')
    if pivot_windows:
        kwargs_list = [w['kwargs'] for w in pivot_windows]
        orders = model._rank_windows_batch(kwargs_list)
        apply_tdpart_batched_results(pivot_windows, orders, queries_state)

    # Phase 2: Candidate growth (iterative with budget filling)
    max_rounds = 100  # Safety limit
    for _ in range(max_rounds):
        grow_windows = collect_tdpart_windows_for_batching(queries_state, 'grow')
        if not grow_windows:
            break  # All queries exhausted or done

        kwargs_list = [w['kwargs'] for w in grow_windows]
        orders = model._rank_windows_batch(kwargs_list)
        apply_tdpart_batched_results(grow_windows, orders, queries_state)

    # Phase 3: Finalize iteration for each query
    active_count = 0
    for state in queries_state.values():
        if state['phase'] != 'done':
            check_and_update_query_state(state, iteration)
            if state['phase'] != 'done':
                active_count += 1

    return active_count


def tdpart_final_collation_batched(model, queries_state):
    """
    Perform final ranking of candidates to determine top-k across all queries.

    For queries where len(candidates) > cutoff, we need to re-rank all candidates
    to get the definitive top-k ordering, since multiple docs may have beaten the
    pivot at various points.

    Args:
        model: The generative ranker model
        queries_state: Dict mapping qid -> query state

    Modifies queries_state in-place with final rankings.
    """
    # Collect final ranking windows for queries that need it
    final_windows = []
    for qid, state in queries_state.items():
        if state['phase'] == 'done':
            candidates = state['candidates']
            cutoff = state['cutoff']

            # Only re-rank if we have more candidates than cutoff
            if len(candidates) > cutoff:
                final_windows.append({
                    'qid': qid,
                    'window': candidates,
                    'kwargs': {
                        'qid': qid,
                        'query': state['query'],
                        'doc_text': candidates.doc_texts.tolist(),
                        'doc_idx': candidates.doc_idx.tolist(),
                        'start_idx': 0,
                        'end_idx': len(candidates),
                        'window_len': len(candidates)
                    },
                    'type': 'final_collation'
                })

    # Batch rank all final windows
    if final_windows:
        kwargs_list = [w['kwargs'] for w in final_windows]
        orders = model._rank_windows_batch(kwargs_list)

        # Apply final rankings
        for window_data, order in zip(final_windows, orders):
            qid = window_data['qid']
            state = queries_state[qid]
            candidates = window_data['window']

            # Apply ordering to get final ranked candidates
            sorted_candidates = apply_order(candidates, order)

            # Update state with re-ranked candidates
            state['candidates'] = sorted_candidates


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
    b = l[pivot_pos + 1:]           # backfill worse than pivot (in current view)

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
        b = b + l2[p_idx + 1:]

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


__all__ = [
    'tdpart',
    'initialize_tdpart_queries',
    'tdpart_batched_iteration',
    'tdpart_final_collation_batched',
    'convert_tdpart_states_to_windows',
]
