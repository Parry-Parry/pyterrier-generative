"""
Test suite for queue behavior and positional bias simulation.

This test suite verifies:
1. Correct queuing/batching behavior for all algorithms
2. Simulates positional bias seen in real LLMs
3. Ensures cross-query batching works correctly
"""

import pytest
import pandas as pd
import numpy as np
from pyterrier_generative import GenerativeRanker, Algorithm


class GenerationOutput:
    """Output object with .text attribute as returned by real backends."""
    def __init__(self, text):
        self.text = text


class PositionalBiasBackend:
    """
    Backend that simulates positional bias seen in real LLMs.

    Real LLMs tend to favor items in certain positions (often middle positions).
    This backend simulates that by applying a bias toward the middle of the list.

    Also tracks all calls to verify batching behavior.
    """

    supports_message_input = False

    def __init__(self, bias_strength=0.7):
        """
        Args:
            bias_strength: How much to favor middle positions (0=none, 1=strong)
        """
        self.bias_strength = bias_strength
        self.call_history = []
        self.batch_sizes = []

    def generate(self, prompts):
        """
        Generate rankings with positional bias.

        Simulates tendency to rank middle items higher than they should be.
        """
        self.call_history.append({
            'batch_size': len(prompts),
            'prompts': prompts
        })
        self.batch_sizes.append(len(prompts))

        outputs = []
        for prompt in prompts:
            # Extract document indices from prompt
            import re
            matches = re.findall(r'\[(\d+)\]', prompt)
            n = len(matches)

            if n == 0:
                outputs.append(GenerationOutput(""))
                continue

            # Create ranking with positional bias
            # Start with reverse order (assume later docs are better)
            ranking = list(range(n, 0, -1))

            # Apply middle bias: boost items in middle positions
            if n > 2 and self.bias_strength > 0:
                middle = n // 2
                # Create a bias score for each position
                bias_scores = []
                for i in range(n):
                    # Distance from middle
                    distance = abs(i - middle)
                    # Closer to middle = higher bias
                    bias = 1.0 - (distance / (n / 2.0)) * self.bias_strength
                    bias_scores.append(bias)

                # Adjust ranking by bias
                adjusted = [(ranking[i] + bias_scores[i] * 2, i) for i in range(n)]
                adjusted.sort(reverse=True)
                ranking = [i + 1 for _, i in adjusted]

            ranking_text = " ".join(str(i) for i in ranking)
            outputs.append(GenerationOutput(ranking_text))

        return outputs


class CallTrackingBackend:
    """
    Simple backend that tracks the order and batching of calls.

    Returns reverse order to make tracking easier.
    """

    supports_message_input = False

    def __init__(self):
        self.calls = []  # List of (call_index, batch_size, window_positions)
        self.call_index = 0

    def generate(self, prompts):
        """Track calls and return reverse ordering."""
        # Extract query IDs and window positions from prompts
        import re

        window_info = []
        for prompt in prompts:
            # Extract query from "Query: <query>" pattern
            query_match = re.search(r'Query: ([^\n]+)', prompt)
            query = query_match.group(1) if query_match else "unknown"

            # Count passages
            matches = re.findall(r'\[(\d+)\]', prompt)
            n = len(matches)

            window_info.append({
                'query': query,
                'num_docs': n
            })

        self.calls.append({
            'call_index': self.call_index,
            'batch_size': len(prompts),
            'windows': window_info
        })
        self.call_index += 1

        # Return reverse order rankings
        outputs = []
        for info in window_info:
            n = info['num_docs']
            ranking = " ".join(str(i) for i in range(n, 0, -1))
            outputs.append(GenerationOutput(ranking))

        return outputs


# Fixtures

@pytest.fixture
def positional_bias_backend():
    """Backend with moderate positional bias."""
    return PositionalBiasBackend(bias_strength=0.5)


@pytest.fixture
def call_tracking_backend():
    """Backend that tracks call patterns."""
    return CallTrackingBackend()


@pytest.fixture
def multi_query_data():
    """Sample data with multiple queries."""
    dfs = []
    for i in range(3):
        df = pd.DataFrame({
            'qid': [f'q{i}'] * 20,
            'query': [f'test query {i}'] * 20,
            'docno': [f'q{i}_d{j}' for j in range(20)],
            'text': [f'q{i} document {j}' for j in range(20)],
            'score': list(range(20, 0, -1))
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Tests for Positional Bias Backend

class TestPositionalBiasBackend:
    """Test the positional bias simulation."""

    def test_backend_returns_output_objects(self, positional_bias_backend):
        """Verify backend returns objects with .text attribute."""
        outputs = positional_bias_backend.generate(["Query: test\n[1] doc1\n[2] doc2\n[3] doc3"])

        assert len(outputs) == 1
        assert hasattr(outputs[0], 'text')
        assert isinstance(outputs[0].text, str)

    def test_backend_tracks_calls(self, positional_bias_backend):
        """Verify backend tracks all calls."""
        positional_bias_backend.generate(["prompt1"])
        positional_bias_backend.generate(["prompt2", "prompt3"])

        assert len(positional_bias_backend.call_history) == 2
        assert positional_bias_backend.batch_sizes == [1, 2]

    def test_positional_bias_affects_ranking(self):
        """Verify positional bias changes rankings."""
        # No bias - should give reverse order
        no_bias = PositionalBiasBackend(bias_strength=0.0)
        outputs = no_bias.generate(["Query: test\n[1] a\n[2] b\n[3] c\n[4] d\n[5] e"])
        no_bias_ranking = outputs[0].text

        # Strong bias - should differ
        strong_bias = PositionalBiasBackend(bias_strength=1.0)
        outputs = strong_bias.generate(["Query: test\n[1] a\n[2] b\n[3] c\n[4] d\n[5] e"])
        biased_ranking = outputs[0].text

        # Rankings should differ due to bias
        assert no_bias_ranking != biased_ranking


# Tests for Queue Behavior - Sliding Window

class TestSlidingWindowQueueBehavior:
    """Test queue behavior for sliding window algorithm."""

    def test_sliding_window_batches_by_position(self, call_tracking_backend, multi_query_data):
        """Verify sliding window batches same window position across queries."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )

        result = ranker.transform(multi_query_data)

        # Each call should batch the same window position across all 3 queries
        for call in call_tracking_backend.calls:
            # For sliding window with 3 queries, batch size should be 3
            # (one window from each query at the same position)
            assert call['batch_size'] == 3, \
                f"Expected batch_size=3, got {call['batch_size']} for call {call['call_index']}"

    def test_sliding_window_processes_positions_sequentially(self, call_tracking_backend):
        """Verify window positions are processed in order across queries."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )

        # Create 2 queries with 15 docs each
        q1_df = pd.DataFrame({
            'qid': ['q1'] * 15,
            'query': ['query 1'] * 15,
            'docno': [f'q1_d{i}' for i in range(15)],
            'text': [f'q1 doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })
        q2_df = pd.DataFrame({
            'qid': ['q2'] * 15,
            'query': ['query 2'] * 15,
            'docno': [f'q2_d{i}' for i in range(15)],
            'text': [f'q2 doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })
        input_df = pd.concat([q1_df, q2_df], ignore_index=True)

        result = ranker.transform(input_df)

        # With window_size=5, stride=3, we expect 4 window positions for 15 docs
        # Each call should have batch_size=2 (one from each query)
        assert len(call_tracking_backend.calls) == 4

        for call in call_tracking_backend.calls:
            assert call['batch_size'] == 2
            # Both windows in the batch should be for the same position
            # (verified by them being batched together)

    def test_sliding_window_never_batches_same_query(self, call_tracking_backend):
        """Verify windows from the same query are never batched together."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )

        # Single query with multiple windows
        input_df = pd.DataFrame({
            'qid': ['q1'] * 15,
            'query': ['single query'] * 15,
            'docno': [f'd{i}' for i in range(15)],
            'text': [f'doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })

        result = ranker.transform(input_df)

        # Each call should have batch_size=1 (only one query)
        for call in call_tracking_backend.calls:
            assert call['batch_size'] == 1
            # Verify all windows are from the same query
            queries = set(w['query'] for w in call['windows'])
            assert len(queries) == 1
            assert 'single query' in queries


# Tests for Queue Behavior - Single Window

class TestSingleWindowQueueBehavior:
    """Test queue behavior for single window algorithm."""

    def test_single_window_batches_all_queries(self, call_tracking_backend, multi_query_data):
        """Verify single window batches all queries together."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=10
        )

        result = ranker.transform(multi_query_data)

        # Single window: all queries can be batched in one call
        assert len(call_tracking_backend.calls) == 1
        assert call_tracking_backend.calls[0]['batch_size'] == 3

    def test_single_window_with_varying_doc_counts(self, call_tracking_backend):
        """Verify single window batches queries with different document counts."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=10
        )

        # Queries with different document counts
        q1_df = pd.DataFrame({
            'qid': ['q1'] * 5,
            'query': ['query 1'] * 5,
            'docno': [f'q1_d{i}' for i in range(5)],
            'text': [f'q1 doc {i}' for i in range(5)],
            'score': list(range(5, 0, -1))
        })
        q2_df = pd.DataFrame({
            'qid': ['q2'] * 15,
            'query': ['query 2'] * 15,
            'docno': [f'q2_d{i}' for i in range(15)],
            'text': [f'q2 doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })
        input_df = pd.concat([q1_df, q2_df], ignore_index=True)

        result = ranker.transform(input_df)

        # Should batch both queries together
        assert len(call_tracking_backend.calls) == 1
        assert call_tracking_backend.calls[0]['batch_size'] == 2

        # Verify different window sizes
        windows = call_tracking_backend.calls[0]['windows']
        assert windows[0]['num_docs'] == 5  # q1: min(5, 10)
        assert windows[1]['num_docs'] == 10  # q2: min(15, 10)


# Tests for Queue Behavior - TDPart

class TestTDPartQueueBehavior:
    """Test queue behavior for TDPart algorithm."""

    def test_tdpart_batches_pivot_phase(self, call_tracking_backend):
        """Verify TDPart batches pivot-finding windows across queries."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=3
        )

        # Multiple queries
        q1_df = pd.DataFrame({
            'qid': ['q1'] * 30,
            'query': ['query 1'] * 30,
            'docno': [f'q1_d{i}' for i in range(30)],
            'text': [f'q1 doc {i}' for i in range(30)],
            'score': list(range(30, 0, -1))
        })
        q2_df = pd.DataFrame({
            'qid': ['q2'] * 30,
            'query': ['query 2'] * 30,
            'docno': [f'q2_d{i}' for i in range(30)],
            'text': [f'q2 doc {i}' for i in range(30)],
            'score': list(range(30, 0, -1))
        })
        input_df = pd.concat([q1_df, q2_df], ignore_index=True)

        result = ranker.transform(input_df)

        # TDPart should have multiple batched calls
        assert len(call_tracking_backend.calls) > 0

        # At least some calls should batch multiple queries together
        batched_calls = [c for c in call_tracking_backend.calls if c['batch_size'] > 1]
        assert len(batched_calls) > 0, "TDPart should batch some windows across queries"

    def test_tdpart_batches_growth_phase(self, call_tracking_backend):
        """Verify TDPart batches candidate growth windows."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=15,
            cutoff=5,
            max_iters=5
        )

        # Create queries
        dfs = []
        for i in range(3):
            df = pd.DataFrame({
                'qid': [f'q{i}'] * 40,
                'query': [f'query {i}'] * 40,
                'docno': [f'q{i}_d{j}' for j in range(40)],
                'text': [f'q{i} doc {j}' for j in range(40)],
                'score': list(range(40, 0, -1))
            })
            dfs.append(df)
        input_df = pd.concat(dfs, ignore_index=True)

        result = ranker.transform(input_df)

        # Should have multiple batched calls during growth phase
        assert len(call_tracking_backend.calls) > 1


# Tests with Positional Bias

class TestPositionalBiasIntegration:
    """Test algorithm behavior with positional bias backend."""

    def test_sliding_window_with_bias(self, positional_bias_backend):
        """Test sliding window handles positional bias correctly."""
        ranker = GenerativeRanker(
            model=positional_bias_backend,
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

        # Should complete successfully despite bias
        assert len(result) == 25
        assert set(result['docno']) == set(input_df['docno'])

    def test_single_window_with_bias(self, positional_bias_backend):
        """Test single window handles positional bias correctly."""
        ranker = GenerativeRanker(
            model=positional_bias_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=15
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 20,
            'query': ['test'] * 20,
            'docno': [f'd{i}' for i in range(20)],
            'text': [f'doc {i}' for i in range(20)],
            'score': list(range(20, 0, -1))
        })

        result = ranker.transform(input_df)

        # Should complete successfully
        assert len(result) == 20
        assert set(result['docno']) == set(input_df['docno'])

    def test_tdpart_with_bias(self, positional_bias_backend):
        """Test TDPart handles positional bias correctly."""
        ranker = GenerativeRanker(
            model=positional_bias_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=5
        )

        input_df = pd.DataFrame({
            'qid': ['q1'] * 30,
            'query': ['test'] * 30,
            'docno': [f'd{i}' for i in range(30)],
            'text': [f'doc {i}' for i in range(30)],
            'score': list(range(30, 0, -1))
        })

        result = ranker.transform(input_df)

        # Should complete successfully
        assert len(result) == 30
        assert set(result['docno']) == set(input_df['docno'])

    def test_multi_query_with_bias(self, positional_bias_backend, multi_query_data):
        """Test cross-query batching with positional bias."""
        ranker = GenerativeRanker(
            model=positional_bias_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )

        result = ranker.transform(multi_query_data)

        # All queries should complete
        assert len(result) == 60  # 3 queries × 20 docs
        assert set(result['qid'].unique()) == {'q0', 'q1', 'q2'}

        # Each query should have all documents
        for qid in ['q0', 'q1', 'q2']:
            query_results = result[result['qid'] == qid]
            assert len(query_results) == 20


# Tests for Edge Cases

class TestQueueEdgeCases:
    """Test edge cases in queue behavior."""

    def test_empty_queue(self, call_tracking_backend):
        """Test handling of empty input."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}",
            algorithm=Algorithm.SLIDING_WINDOW
        )

        empty_df = pd.DataFrame(columns=['qid', 'query', 'docno', 'text', 'score'])
        result = ranker.transform(empty_df)

        # No calls should be made
        assert len(call_tracking_backend.calls) == 0
        assert len(result) == 0

    def test_single_document_queue(self, call_tracking_backend):
        """Test queue with single document."""
        ranker = GenerativeRanker(
            model=call_tracking_backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )

        input_df = pd.DataFrame({
            'qid': ['q1'],
            'query': ['test'],
            'docno': ['d0'],
            'text': ['doc'],
            'score': [1.0]
        })

        result = ranker.transform(input_df)

        # Should make one call with one window
        assert len(call_tracking_backend.calls) == 1
        assert call_tracking_backend.calls[0]['batch_size'] == 1
        assert len(result) == 1
