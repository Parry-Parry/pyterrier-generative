"""
GPU equivalence tests to verify batched and non-batched rankings are identical.

This test suite ensures that:
1. Mocked tests: Rankings are COMPLETELY IDENTICAL (bit-for-bit)
2. Real LLM tests: Rankings are NEAR IDENTICAL (within tolerance)

Covers all algorithms: SLIDING_WINDOW, SINGLE_WINDOW, TDPART, SETWISE

Supports both CUDA (NVIDIA) and MPS (Apple Silicon) GPUs.
"""

import re
import pytest
import pandas as pd

from pyterrier_generative import GenerativeRanker, Algorithm


# =============================================================================
# GPU Detection (CUDA and MPS)
# =============================================================================

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = (
        torch.backends.mps.is_available()
        if hasattr(torch.backends, 'mps') else False
    )
    GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE

    if CUDA_AVAILABLE:
        GPU_TYPE = "cuda"
        GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
    elif MPS_AVAILABLE:
        GPU_TYPE = "mps"
        GPU_DEVICE_NAME = "Apple Silicon (MPS)"
        # MPS doesn't expose memory info directly, assume sufficient
        GPU_MEMORY_GB = 8.0
    else:
        GPU_TYPE = None
        GPU_DEVICE_NAME = None
        GPU_MEMORY_GB = 0
except ImportError:
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_TYPE = None
    GPU_DEVICE_NAME = None
    GPU_MEMORY_GB = 0


# =============================================================================
# Test Infrastructure
# =============================================================================

class GenerationOutput:
    """Wrapper to match real backend output format."""
    def __init__(self, text):
        self.text = text


class DeterministicBackend:
    """
    Deterministic backend that produces identical outputs regardless of batch.

    This backend is used to verify that batched and non-batched processing
    produces EXACTLY the same results. It returns reverse order rankings
    (N, N-1, ..., 1) based on the number of passages in the prompt.
    """

    supports_message_input = False

    def __init__(self):
        self.call_history = []
        self.batch_calls = 0
        self.single_calls = 0

    def generate(self, prompts):
        """Generate deterministic rankings."""
        batch_size = len(prompts)
        if batch_size > 1:
            self.batch_calls += 1
        else:
            self.single_calls += 1

        self.call_history.append({
            'batch_size': batch_size,
            'prompts': prompts
        })

        outputs = []
        for prompt in prompts:
            # Count passages in prompt by finding [N] markers
            matches = re.findall(r'\[(\d+)\]', prompt)
            n = len(matches)

            # Return reverse order: N, N-1, ..., 1
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
            'text': [f'q{i} document {j} content' for j in range(docs_per_query)],
            'score': list(range(docs_per_query, 0, -1))
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def create_varying_doc_count_data():
    """Create test data with varying document counts per query."""
    dfs = []
    doc_counts = [10, 25, 15, 30, 8]
    for i, count in enumerate(doc_counts):
        df = pd.DataFrame({
            'qid': [f'q{i}'] * count,
            'query': [f'test query {i}'] * count,
            'docno': [f'q{i}_d{j}' for j in range(count)],
            'text': [f'q{i} document {j} content' for j in range(count)],
            'score': list(range(count, 0, -1))
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def normalize_results(df):
    """Normalize results for comparison (sort by qid, then rank)."""
    return df.sort_values(['qid', 'rank']).reset_index(drop=True)


def verify_exact_equivalence(individual_df, batched_df, algorithm_name):
    """
    Verify that two result DataFrames are COMPLETELY IDENTICAL.

    Checks:
    1. Same number of results per query
    2. Exact same docno ordering per query
    3. Exact same rank values per query
    """
    individual_norm = normalize_results(individual_df)
    batched_norm = normalize_results(batched_df)

    # Check total count
    assert len(individual_norm) == len(batched_norm), (
        f"{algorithm_name}: Different result counts "
        f"(individual={len(individual_norm)}, batched={len(batched_norm)})"
    )

    # Check per-query equivalence
    for qid in individual_norm['qid'].unique():
        ind_q = individual_norm[individual_norm['qid'] == qid]
        bat_q = batched_norm[batched_norm['qid'] == qid]

        # Count check
        assert len(ind_q) == len(bat_q), (
            f"{algorithm_name} qid={qid}: Different document counts "
            f"(individual={len(ind_q)}, batched={len(bat_q)})"
        )

        # Exact docno order
        ind_docnos = list(ind_q['docno'])
        bat_docnos = list(bat_q['docno'])
        assert ind_docnos == bat_docnos, (
            f"{algorithm_name} qid={qid}: Different document ordering\n"
            f"Individual: {ind_docnos[:10]}...\n"
            f"Batched: {bat_docnos[:10]}..."
        )

        # Exact rank values
        assert list(ind_q['rank']) == list(bat_q['rank']), \
            f"{algorithm_name} qid={qid}: Different rank values"


def verify_near_identical(individual_df, batched_df,
                          kendall_threshold=0.95,
                          spearman_threshold=0.95,
                          max_position_drift=3):
    """
    Verify rankings are near-identical within tolerance.

    Used for real LLM tests where floating-point variations may occur.

    Returns:
        dict with 'passed' bool and 'details' list
    """
    from scipy.stats import kendalltau, spearmanr

    results = {'passed': True, 'details': []}

    individual_norm = normalize_results(individual_df)
    batched_norm = normalize_results(batched_df)

    for qid in individual_norm['qid'].unique():
        ind_q = individual_norm[individual_norm['qid'] == qid]
        ind_q = ind_q.sort_values('rank')
        bat_q = batched_norm[batched_norm['qid'] == qid]
        bat_q = bat_q.sort_values('rank')

        # Create position mappings
        ind_positions = {doc: i for i, doc in enumerate(ind_q['docno'])}
        bat_positions = {doc: i for i, doc in enumerate(bat_q['docno'])}

        # Calculate correlation metrics
        common_docs = set(ind_positions.keys()) & set(bat_positions.keys())
        if len(common_docs) < 2:
            continue

        ind_ranks = [ind_positions[d] for d in common_docs]
        bat_ranks = [bat_positions[d] for d in common_docs]

        tau, _ = kendalltau(ind_ranks, bat_ranks)
        rho, _ = spearmanr(ind_ranks, bat_ranks)

        # Check position drift
        max_drift = max(
            abs(ind_positions[d] - bat_positions[d]) for d in common_docs
        )

        query_passed = (
            tau >= kendall_threshold and
            rho >= spearman_threshold and
            max_drift <= max_position_drift
        )

        results['details'].append({
            'qid': qid,
            'kendall_tau': tau,
            'spearman_rho': rho,
            'max_drift': max_drift,
            'passed': query_passed
        })

        if not query_passed:
            results['passed'] = False

    return results


# =============================================================================
# Standard Prompt Template
# =============================================================================

STANDARD_PROMPT = (
    "Query: {{ query }}\n"
    "{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}"
)


# =============================================================================
# Mocked Tests - SLIDING_WINDOW
# =============================================================================

@pytest.mark.gpu
class TestGPUMockedSlidingWindow:
    """Test SLIDING_WINDOW batched vs non-batched equivalence."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("window_size,stride", [
        (5, 3),
        (10, 5),
        (15, 7),
        (20, 10),
    ])
    def test_sliding_window_exact_equivalence(self, window_size, stride):
        """Verify batched sliding window produces IDENTICAL results."""
        input_df = create_test_data(num_queries=3, docs_per_query=20)

        # Process queries individually (no cross-query batching)
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
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
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=window_size,
            stride=stride
        )
        batched_result = batched_ranker.transform(input_df)

        # Verify exact equivalence
        verify_exact_equivalence(
            individual_combined, batched_result, "SLIDING_WINDOW"
        )

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_sliding_window_varying_doc_counts(self):
        """Verify batching works with varying document counts."""
        input_df = create_varying_doc_count_data()

        # Process individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=10,
                stride=5
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process batched
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        batched_result = batched_ranker.transform(input_df)

        verify_exact_equivalence(
            individual_combined, batched_result, "SLIDING_WINDOW"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_sliding_window_single_query(self):
        """Verify batching works correctly with a single query."""
        input_df = create_test_data(num_queries=1, docs_per_query=25)

        backend1 = DeterministicBackend()
        ranker1 = GenerativeRanker(
            model=backend1,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        result1 = ranker1.transform(input_df)

        backend2 = DeterministicBackend()
        ranker2 = GenerativeRanker(
            model=backend2,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        result2 = ranker2.transform(input_df)

        # Results should be identical across runs
        verify_exact_equivalence(
            result1, result2, "SLIDING_WINDOW single query"
        )


# =============================================================================
# Mocked Tests - SINGLE_WINDOW
# =============================================================================

@pytest.mark.gpu
class TestGPUMockedSingleWindow:
    """Test SINGLE_WINDOW batched vs non-batched equivalence."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("window_size", [10, 15, 20, 25])
    def test_single_window_exact_equivalence(self, window_size):
        """Verify batched single window produces IDENTICAL results."""
        input_df = create_test_data(num_queries=3, docs_per_query=30)

        # Process queries individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
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
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=window_size
        )
        batched_result = batched_ranker.transform(input_df)

        # Verify exact equivalence
        verify_exact_equivalence(
            individual_combined, batched_result, "SINGLE_WINDOW"
        )

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_single_window_varying_doc_counts(self):
        """Verify batching works with varying document counts."""
        input_df = create_varying_doc_count_data()

        # Process individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SINGLE_WINDOW,
                window_size=10
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process batched
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=10
        )
        batched_result = batched_ranker.transform(input_df)

        verify_exact_equivalence(
            individual_combined, batched_result, "SINGLE_WINDOW"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_single_window_many_queries(self):
        """Verify batching handles many queries efficiently."""
        input_df = create_test_data(num_queries=10, docs_per_query=15)

        # Process individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SINGLE_WINDOW,
                window_size=10
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process batched
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=10
        )
        batched_result = batched_ranker.transform(input_df)

        verify_exact_equivalence(
            individual_combined, batched_result, "SINGLE_WINDOW"
        )

        # Verify batching was efficient (single batch call)
        assert batched_backend.batch_calls == 1, \
            "Should batch all windows in single call"


# =============================================================================
# Mocked Tests - TDPART
# =============================================================================

@pytest.mark.gpu
class TestGPUMockedTDPart:
    """Test TDPART batched vs non-batched equivalence."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("window_size,buffer,cutoff", [
        (10, 10, 5),
        (10, 15, 8),
        (15, 20, 10),
        (20, 25, 12),
    ])
    def test_tdpart_exact_equivalence(self, window_size, buffer, cutoff):
        """Verify batched TDPart produces IDENTICAL results."""
        input_df = create_test_data(num_queries=3, docs_per_query=30)

        # Process queries individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
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
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=window_size,
            buffer=buffer,
            cutoff=cutoff,
            max_iters=10
        )
        batched_result = batched_ranker.transform(input_df)

        # Verify exact equivalence
        verify_exact_equivalence(
            individual_combined, batched_result, "TDPART"
        )

        # Verify batching was actually used
        assert batched_backend.batch_calls > 0, "Batching was not used"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_tdpart_varying_doc_counts(self):
        """Verify batching works with varying document counts."""
        input_df = create_varying_doc_count_data()

        # Process individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.TDPART,
                window_size=10,
                buffer=10,
                cutoff=5,
                max_iters=10
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process batched
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        batched_result = batched_ranker.transform(input_df)

        verify_exact_equivalence(
            individual_combined, batched_result, "TDPART"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_tdpart_single_query(self):
        """Verify TDPart works correctly with a single query."""
        input_df = create_test_data(num_queries=1, docs_per_query=30)

        backend1 = DeterministicBackend()
        ranker1 = GenerativeRanker(
            model=backend1,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        result1 = ranker1.transform(input_df)

        backend2 = DeterministicBackend()
        ranker2 = GenerativeRanker(
            model=backend2,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        result2 = ranker2.transform(input_df)

        # Results should be identical across runs
        verify_exact_equivalence(result1, result2, "TDPART single query")


# =============================================================================
# Mocked Tests - SETWISE
# =============================================================================

@pytest.mark.gpu
class TestGPUMockedSetwise:
    """
    Test SETWISE algorithm consistency.

    IMPORTANT: SETWISE does NOT support cross-query batching.
    These tests verify consistent results across runs.
    """

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.parametrize("k", [3, 5, 10])
    def test_setwise_no_cross_query_batching(self, k):
        """Verify SETWISE does not use cross-query batching."""
        input_df = create_test_data(num_queries=1, docs_per_query=15)

        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SETWISE,
            k=k
        )
        ranker.transform(input_df)

        # SETWISE should not use cross-query batching
        # (each call is for a single pairwise comparison)
        assert backend.batch_calls == 0, (
            f"SETWISE should not use cross-query batching, "
            f"but had {backend.batch_calls} batch calls"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_setwise_consistency(self):
        """Verify SETWISE produces consistent results across runs."""
        input_df = create_test_data(num_queries=1, docs_per_query=15)

        backend1 = DeterministicBackend()
        ranker1 = GenerativeRanker(
            model=backend1,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SETWISE,
            k=5
        )
        result1 = ranker1.transform(input_df)

        backend2 = DeterministicBackend()
        ranker2 = GenerativeRanker(
            model=backend2,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SETWISE,
            k=5
        )
        result2 = ranker2.transform(input_df)

        # Results should be identical across runs (deterministic)
        verify_exact_equivalence(result1, result2, "SETWISE")

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_setwise_multi_query_consistency(self):
        """Verify SETWISE produces consistent results with multiple queries."""
        input_df = create_test_data(num_queries=3, docs_per_query=10)

        # Process individually
        individual_results = []
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SETWISE,
                k=5
            )
            query_df = input_df[input_df['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SETWISE,
            k=5
        )
        batched_result = batched_ranker.transform(input_df)

        # Results should be identical (SETWISE processes sequentially)
        verify_exact_equivalence(
            individual_combined, batched_result, "SETWISE"
        )

        # Verify no batching was used
        assert batched_backend.batch_calls == 0, \
            "SETWISE should not use batching"


# =============================================================================
# Real LLM Tests
# =============================================================================

@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
class TestGPURealLLMEquivalence:
    """
    Test with real LLMs that rankings are NEAR IDENTICAL.

    These tests use small LLMs and verify that batched vs non-batched
    processing produces rankings within acceptable tolerance thresholds.

    Tolerance thresholds:
    - Kendall Tau correlation >= 0.95
    - Spearman correlation >= 0.95
    - Max position drift <= 3 positions
    """

    TOLERANCE_KENDALL_TAU = 0.95
    TOLERANCE_SPEARMAN = 0.95
    MAX_POSITION_DRIFT = 3

    @pytest.fixture
    def small_test_data(self):
        """Small dataset for real LLM tests."""
        return create_test_data(num_queries=2, docs_per_query=10)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.skipif(
        GPU_MEMORY_GB < 4,
        reason="Insufficient GPU memory (need >= 4GB)"
    )
    def test_real_llm_sliding_window(self, small_test_data):
        """Test real LLM sliding window batched vs non-batched."""
        try:
            from pyterrier_rag.backend import HuggingFaceBackend
        except ImportError:
            pytest.skip("HuggingFaceBackend not available")

        try:
            # Use a small model for testing
            backend_individual = HuggingFaceBackend(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=50
            )
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

        # Process queries individually
        individual_results = []
        for qid in small_test_data['qid'].unique():
            ranker = GenerativeRanker(
                model=backend_individual,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=5,
                stride=3
            )
            query_df = small_test_data[small_test_data['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with batching)
        backend_batched = HuggingFaceBackend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=50
        )
        batched_ranker = GenerativeRanker(
            model=backend_batched,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )
        batched_result = batched_ranker.transform(small_test_data)

        # Verify near-identical results
        results = verify_near_identical(
            individual_combined,
            batched_result,
            kendall_threshold=self.TOLERANCE_KENDALL_TAU,
            spearman_threshold=self.TOLERANCE_SPEARMAN,
            max_position_drift=self.MAX_POSITION_DRIFT
        )

        assert results['passed'], (
            f"Real LLM SLIDING_WINDOW results not within tolerance:\n"
            f"{results['details']}"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.skipif(
        GPU_MEMORY_GB < 4,
        reason="Insufficient GPU memory (need >= 4GB)"
    )
    def test_real_llm_single_window(self, small_test_data):
        """Test real LLM single window batched vs non-batched."""
        try:
            from pyterrier_rag.backend import HuggingFaceBackend
        except ImportError:
            pytest.skip("HuggingFaceBackend not available")

        try:
            backend_individual = HuggingFaceBackend(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=50
            )
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

        # Process queries individually
        individual_results = []
        for qid in small_test_data['qid'].unique():
            ranker = GenerativeRanker(
                model=backend_individual,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SINGLE_WINDOW,
                window_size=8
            )
            query_df = small_test_data[small_test_data['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with batching)
        backend_batched = HuggingFaceBackend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=50
        )
        batched_ranker = GenerativeRanker(
            model=backend_batched,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=8
        )
        batched_result = batched_ranker.transform(small_test_data)

        # Verify near-identical results
        results = verify_near_identical(
            individual_combined,
            batched_result,
            kendall_threshold=self.TOLERANCE_KENDALL_TAU,
            spearman_threshold=self.TOLERANCE_SPEARMAN,
            max_position_drift=self.MAX_POSITION_DRIFT
        )

        assert results['passed'], (
            f"Real LLM SINGLE_WINDOW results not within tolerance:\n"
            f"{results['details']}"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.skipif(
        GPU_MEMORY_GB < 4,
        reason="Insufficient GPU memory (need >= 4GB)"
    )
    def test_real_llm_tdpart(self, small_test_data):
        """Test real LLM TDPart batched vs non-batched."""
        try:
            from pyterrier_rag.backend import HuggingFaceBackend
        except ImportError:
            pytest.skip("HuggingFaceBackend not available")

        try:
            backend_individual = HuggingFaceBackend(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=50
            )
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

        # Process queries individually
        individual_results = []
        for qid in small_test_data['qid'].unique():
            ranker = GenerativeRanker(
                model=backend_individual,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.TDPART,
                window_size=5,
                buffer=5,
                cutoff=3,
                max_iters=5
            )
            query_df = small_test_data[small_test_data['qid'] == qid]
            result = ranker.transform(query_df)
            individual_results.append(result)
        individual_combined = pd.concat(individual_results, ignore_index=True)

        # Process all queries together (with batching)
        backend_batched = HuggingFaceBackend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=50
        )
        batched_ranker = GenerativeRanker(
            model=backend_batched,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=5,
            buffer=5,
            cutoff=3,
            max_iters=5
        )
        batched_result = batched_ranker.transform(small_test_data)

        # Verify near-identical results
        results = verify_near_identical(
            individual_combined,
            batched_result,
            kendall_threshold=self.TOLERANCE_KENDALL_TAU,
            spearman_threshold=self.TOLERANCE_SPEARMAN,
            max_position_drift=self.MAX_POSITION_DRIFT
        )

        assert results['passed'], (
            f"Real LLM TDPART results not within tolerance:\n"
            f"{results['details']}"
        )


# =============================================================================
# Batching Efficiency Tests
# =============================================================================

@pytest.mark.gpu
class TestBatchingEfficiency:
    """Test that batching reduces the number of backend calls."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_sliding_window_batching_reduces_calls(self):
        """Verify batching reduces backend calls for sliding window."""
        input_df = create_test_data(num_queries=3, docs_per_query=20)

        # Individual processing: count total calls
        individual_total_calls = 0
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
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
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=10,
            stride=5
        )
        batched_ranker.transform(input_df)
        batched_total_calls = len(batched_backend.call_history)

        # Batching should reduce calls
        assert batched_total_calls < individual_total_calls, (
            f"Batching didn't reduce calls: "
            f"individual={individual_total_calls}, batched={batched_total_calls}"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_single_window_batching_reduces_calls(self):
        """Verify batching reduces backend calls for single window."""
        input_df = create_test_data(num_queries=5, docs_per_query=20)

        # Individual processing: 5 separate calls
        individual_total_calls = 0
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
                algorithm=Algorithm.SINGLE_WINDOW,
                window_size=15
            )
            query_df = input_df[input_df['qid'] == qid]
            ranker.transform(query_df)
            individual_total_calls += len(backend.call_history)

        # Batched processing: should be 1 call with batch_size=5
        batched_backend = DeterministicBackend()
        batched_ranker = GenerativeRanker(
            model=batched_backend,
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=15
        )
        batched_ranker.transform(input_df)
        batched_total_calls = len(batched_backend.call_history)

        # For SINGLE_WINDOW with N queries, batched should be 1 call
        assert batched_total_calls == 1, (
            f"SINGLE_WINDOW batching should be 1 call, "
            f"got {batched_total_calls}"
        )
        assert individual_total_calls == 5, (
            f"Individual processing should be 5 calls, "
            f"got {individual_total_calls}"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_tdpart_batching_reduces_calls(self):
        """Verify batching reduces backend calls for TDPart."""
        input_df = create_test_data(num_queries=3, docs_per_query=30)

        # Individual processing
        individual_total_calls = 0
        for qid in input_df['qid'].unique():
            backend = DeterministicBackend()
            ranker = GenerativeRanker(
                model=backend,
                prompt=STANDARD_PROMPT,
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
            prompt=STANDARD_PROMPT,
            algorithm=Algorithm.TDPART,
            window_size=10,
            buffer=10,
            cutoff=5,
            max_iters=10
        )
        batched_ranker.transform(input_df)
        batched_total_calls = len(batched_backend.call_history)

        # Batching should reduce calls
        assert batched_total_calls < individual_total_calls, (
            f"Batching didn't reduce calls: "
            f"individual={individual_total_calls}, batched={batched_total_calls}"
        )
