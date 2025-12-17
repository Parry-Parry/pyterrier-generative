"""
Test suite for ranking algorithms.

These tests use real model instances (no mocking) to ensure
the algorithms work correctly end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
from pyterrier_generative._algorithms import (
    sliding_window,
    single_window,
    tdpart,
    Algorithm,
    RankedList,
    iter_windows,
)


class SimpleRanker:
    """
    Simple deterministic ranker for testing algorithms.
    Always ranks documents in reverse order (last doc becomes first).
    """

    def __init__(self, window_size=10, stride=5, buffer=10, cutoff=5, k=5, max_iters=10):
        self.window_size = window_size
        self.stride = stride
        self.buffer = buffer
        self.cutoff = cutoff
        self.k = k
        self.max_iters = max_iters

    def __call__(self, **kwargs):
        """Return reverse ordering."""
        doc_texts = kwargs.get('doc_text', [])
        window_len = kwargs.get('window_len', len(doc_texts))
        # Return reverse order
        return list(range(window_len - 1, -1, -1))


class BatchedRanker(SimpleRanker):
    """
    Ranker that supports batched operations.
    Tracks batch sizes for verification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_sizes = []

    def _rank_windows_batch(self, windows_kwargs):
        """Batch ranking - tracks batch size."""
        self.batch_sizes.append(len(windows_kwargs))
        return [self(**kwargs) for kwargs in windows_kwargs]


# Fixtures

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with 30 documents."""
    return pd.DataFrame({
        'qid': ['q1'] * 30,
        'query': ['test query'] * 30,
        'docno': [f'd{i}' for i in range(30)],
        'text': [f'document {i} content' for i in range(30)],
        'score': list(range(30, 0, -1))
    })


@pytest.fixture
def small_dataframe():
    """Create a small DataFrame with 5 documents."""
    return pd.DataFrame({
        'qid': ['q1'] * 5,
        'query': ['test query'] * 5,
        'docno': [f'd{i}' for i in range(5)],
        'text': [f'document {i} content' for i in range(5)],
        'score': list(range(5, 0, -1))
    })


# Tests for RankedList

class TestRankedList:
    """Test RankedList data structure."""

    def test_init(self):
        """Test RankedList initialization."""
        rl = RankedList(
            doc_idx=np.array(['d0', 'd1', 'd2']),
            doc_texts=np.array(['text0', 'text1', 'text2'])
        )
        assert len(rl) == 3
        assert list(rl.doc_idx) == ['d0', 'd1', 'd2']
        assert list(rl.doc_texts) == ['text0', 'text1', 'text2']

    def test_empty_init(self):
        """Test empty RankedList."""
        rl = RankedList()
        assert len(rl) == 0

    def test_getitem_slice(self):
        """Test slicing RankedList."""
        rl = RankedList(
            doc_idx=np.array(['d0', 'd1', 'd2', 'd3']),
            doc_texts=np.array(['t0', 't1', 't2', 't3'])
        )
        sliced = rl[1:3]
        assert len(sliced) == 2
        assert list(sliced.doc_idx) == ['d1', 'd2']
        assert list(sliced.doc_texts) == ['t1', 't2']

    def test_getitem_int(self):
        """Test integer indexing."""
        rl = RankedList(
            doc_idx=np.array(['d0', 'd1', 'd2']),
            doc_texts=np.array(['t0', 't1', 't2'])
        )
        item = rl[1]
        assert len(item) == 1
        assert item.doc_idx[0] == 'd1'
        assert item.doc_texts[0] == 't1'

    def test_setitem_slice(self):
        """Test slice assignment."""
        rl = RankedList(
            doc_idx=np.array(['d0', 'd1', 'd2', 'd3']),
            doc_texts=np.array(['t0', 't1', 't2', 't3'])
        )
        new_rl = RankedList(
            doc_idx=np.array(['d10', 'd11']),
            doc_texts=np.array(['t10', 't11'])
        )
        rl[1:3] = new_rl
        assert list(rl.doc_idx) == ['d0', 'd10', 'd11', 'd3']
        assert list(rl.doc_texts) == ['t0', 't10', 't11', 't3']

    def test_add(self):
        """Test concatenation."""
        rl1 = RankedList(
            doc_idx=np.array(['d0', 'd1']),
            doc_texts=np.array(['t0', 't1'])
        )
        rl2 = RankedList(
            doc_idx=np.array(['d2', 'd3']),
            doc_texts=np.array(['t2', 't3'])
        )
        combined = rl1 + rl2
        assert len(combined) == 4
        assert list(combined.doc_idx) == ['d0', 'd1', 'd2', 'd3']
        assert list(combined.doc_texts) == ['t0', 't1', 't2', 't3']


# Tests for iter_windows

class TestIterWindows:
    """Test window iteration utility."""

    def test_basic_windows(self):
        """Test basic window generation."""
        windows = list(iter_windows(n=30, window_size=10, stride=5, verbose=True))
        assert len(windows) > 0
        # Check each window has correct format (start_idx, end_idx, window_len)
        for start, end, wlen in windows:
            assert end - start == wlen
            assert wlen <= 10

    def test_small_input(self):
        """Test with input smaller than window size."""
        windows = list(iter_windows(n=5, window_size=10, stride=5, verbose=True))
        assert len(windows) == 1
        start, end, wlen = windows[0]
        assert start == 0
        assert end == 5
        assert wlen == 5

    def test_exact_window_size(self):
        """Test when input size equals window size."""
        windows = list(iter_windows(n=10, window_size=10, stride=5, verbose=True))
        assert len(windows) >= 1


# Tests for sliding_window algorithm

class TestSlidingWindow:
    """Test sliding window algorithm."""

    def test_basic_sliding_window(self, sample_dataframe):
        """Test sliding window produces valid output."""
        model = SimpleRanker(window_size=10, stride=5)
        doc_idx, doc_texts = sliding_window(model, 'test query', sample_dataframe)

        assert len(doc_idx) == 30
        assert len(doc_texts) == 30
        # Check all original docs are present
        assert set(doc_idx) == set(sample_dataframe['docno'])

    def test_sliding_window_small_input(self, small_dataframe):
        """Test sliding window with small input."""
        model = SimpleRanker(window_size=10, stride=5)
        doc_idx, doc_texts = sliding_window(model, 'test query', small_dataframe)

        assert len(doc_idx) == 5
        assert len(doc_texts) == 5

    def test_sliding_window_sequential_processing(self, sample_dataframe):
        """Test sliding window processes windows sequentially for a single query."""
        # The sliding_window function processes windows one at a time sequentially
        # Windows cannot be precomputed as each depends on the previous window's results
        # Cross-query batching happens at the transform level (not tested here)
        model = SimpleRanker(window_size=10, stride=5)
        doc_idx, doc_texts = sliding_window(model, 'test query', sample_dataframe)

        # Should have processed all documents
        assert len(doc_idx) == 30
        assert len(doc_texts) == 30
        assert set(doc_idx) == set(sample_dataframe['docno'])

    def test_sliding_window_fallback(self, sample_dataframe):
        """Test fallback when batching not available."""
        model = SimpleRanker(window_size=10, stride=5)
        # Ensure no _rank_windows_batch method
        assert not hasattr(model, '_rank_windows_batch')

        doc_idx, doc_texts = sliding_window(model, 'test query', sample_dataframe)

        assert len(doc_idx) == 30
        assert len(doc_texts) == 30


# Tests for single_window algorithm

class TestSingleWindow:
    """Test single window algorithm."""

    def test_basic_single_window(self, sample_dataframe):
        """Test single window algorithm."""
        model = SimpleRanker(window_size=10)
        doc_idx, doc_texts = single_window(model, 'test query', sample_dataframe)

        # Should return all 30 docs
        assert len(doc_idx) == 30
        assert len(doc_texts) == 30
        # First 10 should be reranked, rest in original order
        assert set(doc_idx[:10]) == set(sample_dataframe['docno'][:10])
        assert list(doc_idx[10:]) == list(sample_dataframe['docno'][10:])

    def test_single_window_small_input(self, small_dataframe):
        """Test single window with input smaller than window."""
        model = SimpleRanker(window_size=10)
        doc_idx, doc_texts = single_window(model, 'test query', small_dataframe)

        assert len(doc_idx) == 5
        assert len(doc_texts) == 5


# Tests for tdpart algorithm

class TestTDPart:
    """Test TDPart (top-down partition) algorithm."""

    def test_basic_tdpart(self, sample_dataframe):
        """Test TDPart algorithm."""
        model = SimpleRanker(window_size=10, buffer=10, cutoff=5, max_iters=10)
        doc_idx, doc_texts = tdpart(model, 'test query', sample_dataframe)

        # Should return all docs
        assert len(doc_idx) == 30
        assert len(doc_texts) == 30
        # All original docs should be present
        assert set(doc_idx) == set(sample_dataframe['docno'])

    def test_tdpart_small_input(self, small_dataframe):
        """Test TDPart with small input."""
        model = SimpleRanker(window_size=10, buffer=10, cutoff=3, max_iters=10)
        doc_idx, doc_texts = tdpart(model, 'test query', small_dataframe)

        assert len(doc_idx) == 5
        assert len(doc_texts) == 5

    def test_tdpart_cutoff_as_index(self, sample_dataframe):
        """Test TDPart with cutoff_is_index flag."""
        model = SimpleRanker(window_size=10, buffer=10, cutoff=4, max_iters=10)
        model.cutoff_is_index = True
        doc_idx, doc_texts = tdpart(model, 'test query', sample_dataframe)

        assert len(doc_idx) == 30
        assert len(doc_texts) == 30


# Tests for Algorithm enum

class TestAlgorithmEnum:
    """Test Algorithm enumeration."""

    def test_algorithm_values(self):
        """Test all algorithm enum values."""
        assert Algorithm.SLIDING_WINDOW.value == "sliding_window"
        assert Algorithm.SINGLE_WINDOW.value == "single_window"
        assert Algorithm.SETWISE.value == "setwise"
        assert Algorithm.TDPART.value == "tdpart"

    def test_algorithm_members(self):
        """Test algorithm enum membership."""
        assert len(Algorithm) == 4
        assert Algorithm.SLIDING_WINDOW in Algorithm
        assert Algorithm.SINGLE_WINDOW in Algorithm
        assert Algorithm.SETWISE in Algorithm
        assert Algorithm.TDPART in Algorithm
