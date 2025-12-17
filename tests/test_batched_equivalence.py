"""
Integration tests to verify batched algorithms produce identical results to non-batched versions.

This test suite ensures that cross-query batching optimizations don't change the output,
only the efficiency of processing.
"""

import pytest
import pandas as pd
from pyterrier_generative import GenerativeRanker, Algorithm


class GenerationOutput:
    """Wrapper to match real backend output format."""
    def __init__(self, text):
        self.text = text


class DeterministicBackend:
    """
    Deterministic backend that always produces the same ranking.
    Tracks whether batching was used.
    """

    supports_message_input = False

    def __init__(self):
        self.call_history = []
        self.batch_calls = 0
        self.single_calls = 0

    def generate(self, prompts):
        """Generate deterministic rankings."""
        # Track batch size
        if len(prompts) > 1:
            self.batch_calls += 1
        else:
            self.single_calls += 1

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
            outputs.append(GenerationOutput(ranking))

        return outputs


def create_test_data(num_queries=3, docs_per_query=20):
    """Create test data with multiple queries."""
    dfs = []
    for i in range(num_queries):
        df = pd.DataFrame({
            'qid': [f'q{i}'] * docs_per_query,
            'query': [f'test query {i}'] * docs_per_query,
            'docno': [f'q{i}_d{j}' for j in range(docs_per_query)],
            'text': [f'q{i} doc {j}' for j in range(docs_per_query)],
            'score': list(range(docs_per_query, 0, -1))
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def normalize_results(df):
    """Normalize results for comparison (sort by qid, then rank)."""
    return df.sort_values(['qid', 'rank']).reset_index(drop=True)


class TestSlidingWindowEquivalence:
    """Test sliding window batched vs non-batched equivalence."""

    @pytest.mark.parametrize("window_size,stride", [
        (5, 3),
        (10, 5),
        (15, 7),
    ])
    def test_sliding_window_single_vs_batched(self, window_size, stride):
        """Verify batched sliding window produces same results as processing queries individually."""
        input_df = create_test_data(num_queries=3, docs_per_query=20)

        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Process queries individually (no cross-query batching)
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=window_size,
                stride=stride
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with cross-query batching)
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=window_size,
            stride=stride
        )
        batched_result = batched_ranker.transform(input_df)

        # Normalize and compare
        individual_norm = normalize_results(individual_combined)
        batched_norm = normalize_results(batched_result)

        # Verify same number of results
        assert len(individual_norm) == len(batched_norm), \
            f"Different result counts: individual={len(individual_norm)}, batched={len(batched_norm)}"

        # Verify same documents per query
        for qid in input_df['qid'].unique():
            ind_qid = individual_norm[individual_norm['qid'] == qid]
            bat_qid = batched_norm[batched_norm['qid'] == qid]

            assert len(ind_qid) == len(bat_qid), \
                f"Query {qid}: different counts (individual={len(ind_qid)}, batched={len(bat_qid)})"

            # Compare document order
            assert list(ind_qid['docno']) == list(bat_qid['docno']), \
                f"Query {qid}: different document order"

            # Compare ranks
            assert list(ind_qid['rank']) == list(bat_qid['rank']), \
                f"Query {qid}: different ranks"

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"


class TestSingleWindowEquivalence:
    """Test single window batched vs non-batched equivalence."""

    @pytest.mark.parametrize("window_size", [10, 15, 20])
    def test_single_window_single_vs_batched(self, window_size):
        """Verify batched single window produces same results as processing queries individually."""
        input_df = create_test_data(num_queries=3, docs_per_query=25)

        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Process queries individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.SINGLE_WINDOW,
                window_size=window_size
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with cross-query batching)
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=window_size
        )
        batched_result = batched_ranker.transform(input_df)

        # Normalize and compare
        individual_norm = normalize_results(individual_combined)
        batched_norm = normalize_results(batched_result)

        # Verify same results
        assert len(individual_norm) == len(batched_norm)

        for qid in input_df['qid'].unique():
            ind_qid = individual_norm[individual_norm['qid'] == qid]
            bat_qid = batched_norm[batched_norm['qid'] == qid]

            assert len(ind_qid) == len(bat_qid)
            assert list(ind_qid['docno']) == list(bat_qid['docno']), \
                f"Query {qid}: different document order"
            assert list(ind_qid['rank']) == list(bat_qid['rank'])

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"


class TestTDPartEquivalence:
    """Test TDPart batched vs non-batched equivalence."""

    @pytest.mark.parametrize("window_size,buffer,cutoff", [
        (10, 10, 5),
        (10, 15, 8),
        (15, 20, 10),
    ])
    def test_tdpart_single_vs_batched(self, window_size, buffer, cutoff):
        """Verify batched TDPart produces same results as processing queries individually."""
        input_df = create_test_data(num_queries=3, docs_per_query=30)

        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Process queries individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.TDPART,
                window_size=window_size,
                buffer=buffer,
                cutoff=cutoff,
                max_iters=10
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with cross-query batching)
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.TDPART,
            window_size=window_size,
            buffer=buffer,
            cutoff=cutoff,
            max_iters=10
        )
        batched_result = batched_ranker.transform(input_df)

        # Normalize and compare
        individual_norm = normalize_results(individual_combined)
        batched_norm = normalize_results(batched_result)

        # Verify same number of results
        assert len(individual_norm) == len(batched_norm), \
            f"Different result counts: individual={len(individual_norm)}, batched={len(batched_norm)}"

        # Verify same documents per query
        for qid in input_df['qid'].unique():
            ind_qid = individual_norm[individual_norm['qid'] == qid]
            bat_qid = batched_norm[batched_norm['qid'] == qid]

            assert len(ind_qid) == len(bat_qid), \
                f"Query {qid}: different counts (individual={len(ind_qid)}, batched={len(bat_qid)})"

            # Compare document order
            assert list(ind_qid['docno']) == list(bat_qid['docno']), \
                f"Query {qid}: different document order.\nIndividual: {list(ind_qid['docno'])}\nBatched: {list(bat_qid['docno'])}"

            # Compare ranks
            assert list(ind_qid['rank']) == list(bat_qid['rank']), \
                f"Query {qid}: different ranks"

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"


class TestBatchingEfficiency:
    """Test that batching reduces the number of backend calls."""

    def test_batching_reduces_calls_sliding_window(self):
        """Verify batching reduces backend calls for sliding window."""
        input_df = create_test_data(num_queries=3, docs_per_query=20)
        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Individual processing
        individual_total_calls = 0
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=10,
                stride=5
            )
            query_df = input_df[input_df['qid'] == qid]
            ranker.transform(query_df)
            individual_total_calls += len(backend.call_history)

        # Batched processing
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        batched_ranker.transform(input_df)
        batched_total_calls = len(batched_backend.call_history)

        # Batching should significantly reduce calls
        # (individual processes each query separately, batched combines windows across queries)
        assert batched_total_calls < individual_total_calls, \
            f"Batching didn't reduce calls: individual={individual_total_calls}, batched={batched_total_calls}"

    def test_batching_reduces_calls_tdpart(self):
        """Verify batching reduces backend calls for TDPart."""
        input_df = create_test_data(num_queries=3, docs_per_query=30)
        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Individual processing
        individual_total_calls = 0
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.TDPART,
                window_size=10,
                buffer=10,
                cutoff=5,
                max_iters=10
            )
            query_df = input_df[input_df['qid'] == qid]
            ranker.transform(query_df)
            individual_total_calls += len(backend.call_history)

        # Batched processing
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        batched_ranker.transform(input_df)
        batched_total_calls = len(batched_backend.call_history)

        # Batching should significantly reduce calls
        assert batched_total_calls < individual_total_calls, \
            f"Batching didn't reduce calls: individual={individual_total_calls}, batched={batched_total_calls}"


class TestEdgeCases:
    """Test edge cases for batched equivalence."""

    def test_single_query_batching(self):
        """Verify batching works correctly with a single query (should behave the same)."""
        input_df = create_test_data(num_queries=1, docs_per_query=20)
        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        backend1 = DeterministicBackend()
        ranker1 = GenerativeRanker(
            model=backend1,
            prompt=prompt,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        result1 = ranker1.transform(input_df)

        backend2 = DeterministicBackend()
        ranker2 = GenerativeRanker(
            model=backend2,
            prompt=prompt,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        result2 = ranker2.transform(input_df)

        # Results should be identical
        assert list(result1['docno']) == list(result2['docno'])
        assert list(result1['rank']) == list(result2['rank'])

    def test_varying_document_counts(self):
        """Verify batching works with queries having different document counts."""
        # Create queries with different document counts
        q1_df = pd.DataFrame({
            'qid': ['q1'] * 10,
            'query': ['query 1'] * 10,
            'docno': [f'q1_d{i}' for i in range(10)],
            'text': [f'q1 doc {i}' for i in range(10)],
            'score': list(range(10, 0, -1))
        })
        q2_df = pd.DataFrame({
            'qid': ['q2'] * 25,
            'query': ['query 2'] * 25,
            'docno': [f'q2_d{i}' for i in range(25)],
            'text': [f'q2 doc {i}' for i in range(25)],
            'score': list(range(25, 0, -1))
        })
        q3_df = pd.DataFrame({
            'qid': ['q3'] * 15,
            'query': ['query 3'] * 15,
            'docno': [f'q3_d{i}' for i in range(15)],
            'text': [f'q3 doc {i}' for i in range(15)],
            'score': list(range(15, 0, -1))
        })
        input_df = pd.concat([q1_df, q2_df, q3_df], ignore_index=True)

        prompt = "Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"

        # Process individually
        individual_results = []
        for query_df in [q1_df, q2_df, q3_df]:
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=prompt,
                algorithm=Algorithm.TDPART,
                window_size=10,
                buffer=10,
                cutoff=5,
                max_iters=10
            )
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process batched
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=prompt,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        batched_result = batched_ranker.transform(input_df)

        # Verify results
        individual_norm = normalize_results(individual_combined)
        batched_norm = normalize_results(batched_result)

        assert len(individual_norm) == len(batched_norm)

        for qid in ['q1', 'q2', 'q3']:
            ind_qid = individual_norm[individual_norm['qid'] == qid]
            bat_qid = batched_norm[batched_norm['qid'] == qid]

            assert list(ind_qid['docno']) == list(bat_qid['docno']), \
                f"Query {qid}: different document order"
