"""
Integration tests for pyterrier-generative.

These tests verify end-to-end functionality using real (non-mocked) components.
"""

import pytest
import pandas as pd
import numpy as np
from pyterrier_generative import GenerativeRanker, Algorithm


class SimpleDeterministicBackend:
    """
    Deterministic backend that always produces the same ranking.

    This backend reverses the input order to simulate a model
    that prefers later documents.
    """

    supports_message_input = False

    def __init__(self):
        self.call_history = []

    def generate(self, prompts):
        """Generate deterministic rankings."""
        self.call_history.append({
            'batch_size': len(prompts),
            'prompts': prompts
        })

        outputs = []
        for prompt in prompts:
            # Count passages in prompt
            import re
            matches = re.findall(r'\[(\d+)\]', prompt)
            n = len(matches)

            # Return reverse order: N, N-1, ..., 2, 1
            ranking = " ".join(str(i) for i in range(n, 0, -1))
            outputs.append(ranking)

        return outputs


@pytest.fixture
def simple_backend():
    """Provide a simple deterministic backend."""
    return SimpleDeterministicBackend()


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return pd.DataFrame({
        'qid': ['q1', 'q1', 'q1', 'q1', 'q1',
                'q2', 'q2', 'q2', 'q2', 'q2',
                'q3', 'q3', 'q3', 'q3', 'q3'],
        'query': ['machine learning'] * 5 + ['deep learning'] * 5 + ['neural networks'] * 5,
        'docno': [f'q{q}_d{i}' for q in range(1, 4) for i in range(5)],
        'text': [
            f'Document {i} discusses {topic}'
            for topic in ['machine learning', 'deep learning', 'neural networks']
            for i in range(5)
        ],
        'score': list(range(5, 0, -1)) * 3
    })


class TestBasicIntegration:
    """Basic integration tests."""

    def test_single_query_single_window(self, simple_backend):
        """Test ranking a single query with single window."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 5,
            'query': ['test query'] * 5,
            'docno': [f'd{i}' for i in range(5)],
            'text': [f'document {i}' for i in range(5)],
            'score': [5, 4, 3, 2, 1]
        })

        result = ranker.transform(input_df)

        # Verify output structure
        assert len(result) == 5
        assert list(result.columns) == ['qid', 'query', 'docno', 'text', 'rank', 'score']
        assert all(result['qid'] == 'q1')
        assert all(result['query'] == 'test query')

        # Verify ranking (backend reverses order)
        # Original order: d0, d1, d2, d3, d4
        # Backend returns: 5, 4, 3, 2, 1 (reversed)
        # So new order should be: d4, d3, d2, d1, d0
        assert list(result['docno']) == ['d4', 'd3', 'd2', 'd1', 'd0']

        # Verify ranks are sequential
        assert list(result['rank']) == [0, 1, 2, 3, 4]

        # Verify scores are decreasing
        assert all(result['score'][i] > result['score'][i+1] for i in range(4))

    def test_multiple_queries(self, simple_backend, sample_queries):
        """Test ranking multiple queries."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        result = ranker.transform(sample_queries)

        # Should have results for all queries
        assert len(result) == 15
        assert set(result['qid'].unique()) == {'q1', 'q2', 'q3'}

        # Each query should have 5 results
        for qid in ['q1', 'q2', 'q3']:
            query_results = result[result['qid'] == qid]
            assert len(query_results) == 5
            assert list(query_results['rank']) == [0, 1, 2, 3, 4]

    def test_sliding_window_algorithm(self, simple_backend):
        """Test sliding window algorithm."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 15,
            'query': ['test'] * 15,
            'docno': [f'd{i}' for i in range(15)],
            'text': [f'doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })

        result = ranker.transform(input_df)

        # Should return all docs
        assert len(result) == 15
        assert set(result['docno']) == set(input_df['docno'])

        # Backend should have been called multiple times (multiple windows)
        assert len(simple_backend.call_history) > 0

    def test_tdpart_algorithm(self, simple_backend):
        """Test TDPart algorithm."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 20,
            'query': ['test'] * 20,
            'docno': [f'd{i}' for i in range(20)],
            'text': [f'doc {i}' for i in range(20)],
            'score': list(range(20, 0, -1))
        })

        result = ranker.transform(input_df)

        # Should return all docs
        assert len(result) == 20
        assert set(result['docno']) == set(input_df['docno'])


class TestBatchingIntegration:
    """Integration tests for batching functionality."""

    def test_batching_reduces_calls(self, simple_backend):
        """Test that batching reduces number of backend calls."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 15,
            'query': ['test'] * 15,
            'docno': [f'd{i}' for i in range(15)],
            'text': [f'doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })

        result = ranker.transform(input_df)

        # With batching enabled, should make fewer calls
        # For 15 docs with window=5, stride=3: windows at [0:5], [3:8], [6:11], [9:14], [12:15]
        # That's 5 windows, should be batched into 1 call
        assert len(simple_backend.call_history) == 1
        assert simple_backend.call_history[0]['batch_size'] == 5

    def test_batching_produces_correct_results(self, simple_backend):
        """Test that batched processing produces correct results."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 25,
            'query': ['test'] * 25,
            'docno': [f'd{i}' for i in range(25)],
            'text': [f'doc {i}' for i in range(25)],
            'score': list(range(25, 0, -1))
        })

        result = ranker.transform(input_df)

        # All documents should be present
        assert len(result) == 25
        assert set(result['docno']) == set(input_df['docno'])

        # Should have valid ranks
        assert list(result['rank']) == list(range(25))


class TestPromptVariations:
    """Test different prompt formats."""

    def test_callable_prompt(self, simple_backend):
        """Test using callable prompt."""
        def custom_prompt(query, passages, num):
            lines = [f"Search for: {query}", f"Total: {num}"]
            for i, p in enumerate(passages, 1):
                lines.append(f"[{i}] {p}")
            return "\n".join(lines)

        ranker = GenerativeRanker(
            model=simple_backend,
            prompt=custom_prompt,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 5,
            'query': ['test'] * 5,
            'docno': [f'd{i}' for i in range(5)],
            'text': [f'doc {i}' for i in range(5)],
            'score': [5, 4, 3, 2, 1]
        })

        result = ranker.transform(input_df)

        assert len(result) == 5
        # Check prompt was used
        assert len(simple_backend.call_history) > 0
        prompt = simple_backend.call_history[0]['prompts'][0]
        assert "Search for: test" in prompt
        assert "Total: 5" in prompt

    def test_jinja_prompt_with_custom_variables(self, simple_backend):
        """Test Jinja template with custom variables."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="QUERY: {{ query }}\nCOUNT: {{ num }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 3,
            'query': ['information retrieval'] * 3,
            'docno': ['d0', 'd1', 'd2'],
            'text': ['doc 0', 'doc 1', 'doc 2'],
            'score': [3, 2, 1]
        })

        result = ranker.transform(input_df)

        assert len(result) == 3
        prompt = simple_backend.call_history[0]['prompts'][0]
        assert "QUERY: information retrieval" in prompt
        assert "COUNT: 3" in prompt


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_input(self, simple_backend):
        """Test with empty DataFrame."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        empty_df = pd.DataFrame(columns=['qid', 'query', 'docno', 'text', 'score'])
        result = ranker.transform(empty_df)

        assert len(result) == 0
        assert list(result.columns) == ['qid', 'query', 'docno', 'text', 'rank', 'score']

    def test_single_document(self, simple_backend):
        """Test with single document."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        input_df = pd.DataFrame({
            'qid': ['q1'],
            'query': ['test'],
            'docno': ['d0'],
            'text': ['single doc'],
            'score': [1.0]
        })

        result = ranker.transform(input_df)

        assert len(result) == 1
        assert result['docno'].iloc[0] == 'd0'
        assert result['rank'].iloc[0] == 0

    def test_large_document_set(self, simple_backend):
        """Test with large number of documents."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=20,
            stride=10
        )

        n_docs = 100
        input_df = pd.DataFrame({
            'qid': ['q1'] * n_docs,
            'query': ['test'] * n_docs,
            'docno': [f'd{i}' for i in range(n_docs)],
            'text': [f'document {i}' for i in range(n_docs)],
            'score': list(range(n_docs, 0, -1))
        })

        result = ranker.transform(input_df)

        # All documents should be present
        assert len(result) == n_docs
        assert set(result['docno']) == set(input_df['docno'])
        assert list(result['rank']) == list(range(n_docs))


class TestParameterCombinations:
    """Test various parameter combinations."""

    @pytest.mark.parametrize("window_size,stride", [
        (5, 3),
        (10, 5),
        (20, 10),
        (15, 7),
    ])
    def test_window_stride_combinations(self, simple_backend, window_size, stride):
        """Test different window size and stride combinations."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=window_size,
            stride=stride
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 30,
            'query': ['test'] * 30,
            'docno': [f'd{i}' for i in range(30)],
            'text': [f'doc {i}' for i in range(30)],
            'score': list(range(30, 0, -1))
        })

        result = ranker.transform(input_df)

        assert len(result) == 30
        assert set(result['docno']) == set(input_df['docno'])

    @pytest.mark.parametrize("algorithm", [
        Algorithm.SINGLE_WINDOW,
        Algorithm.SLIDING_WINDOW,
        Algorithm.TDPART,
    ])
    def test_all_algorithms(self, simple_backend, algorithm):
        """Test all available algorithms."""
        ranker = GenerativeRanker(
            model=simple_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=algorithm,
            window_size=10,
            stride=5,
            buffer=10,
            cutoff=5,
            max_iters=10
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 20,
            'query': ['test'] * 20,
            'docno': [f'd{i}' for i in range(20)],
            'text': [f'doc {i}' for i in range(20)],
            'score': list(range(20, 0, -1))
        })

        result = ranker.transform(input_df)

        assert len(result) == 20
        assert set(result['docno']) == set(input_df['docno'])
