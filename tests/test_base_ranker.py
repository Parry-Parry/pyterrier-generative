"""
Test suite for GenerativeRanker base class.

Uses a simple deterministic backend for testing (no mocking).
"""

import pytest
import pandas as pd
import re
from pyterrier_generative.modelling.base import GenerativeRanker
from pyterrier_generative._algorithms import Algorithm


class DeterministicBackend:
    """
    Simple deterministic backend for testing.
    Returns predictable rankings based on document position.
    """

    supports_message_input = False

    def __init__(self, reverse=False):
        """
        Args:
            reverse: If True, reverse the ranking order
        """
        self.reverse = reverse
        self.generate_calls = []

    def generate(self, prompts):
        """
        Generate deterministic rankings.

        Returns ranking as "1 2 3 4 ..." or "N ... 3 2 1"
        """
        self.generate_calls.append(len(prompts))
        outputs = []

        for prompt in prompts:
            # Count how many passages are in the prompt
            # Look for [N] patterns
            matches = re.findall(r'\[(\d+)\]', prompt)
            num_passages = len(matches)

            if self.reverse:
                # Return reverse order
                ranking = " ".join(str(i) for i in range(num_passages, 0, -1))
            else:
                # Return sequential order
                ranking = " ".join(str(i) for i in range(1, num_passages + 1))

            outputs.append(ranking)

        return outputs


class MessageBackend(DeterministicBackend):
    """Backend that supports message input."""

    supports_message_input = True

    def generate(self, prompts):
        """
        Generate deterministic rankings from message format.

        Handles both string prompts and message list format.
        """
        self.generate_calls.append(len(prompts))
        outputs = []

        for prompt in prompts:
            # Convert messages to string if needed
            if isinstance(prompt, list):
                # Extract text from messages
                text = ""
                for msg in prompt:
                    if isinstance(msg, dict) and 'content' in msg:
                        text += msg['content'] + "\n"
                prompt = text

            # Count how many passages are in the prompt
            # Look for [N] patterns
            matches = re.findall(r'\[(\d+)\]', prompt)
            num_passages = len(matches)

            if self.reverse:
                # Return reverse order
                ranking = " ".join(str(i) for i in range(num_passages, 0, -1))
            else:
                # Return sequential order
                ranking = " ".join(str(i) for i in range(1, num_passages + 1))

            outputs.append(ranking)

        return outputs


# Fixtures

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'qid': ['q1'] * 10,
        'query': ['test query'] * 10,
        'docno': [f'd{i}' for i in range(10)],
        'text': [f'document {i} content' for i in range(10)],
        'score': list(range(10, 0, -1))
    })


@pytest.fixture
def multi_query_data():
    """Create DataFrame with multiple queries."""
    data = []
    for qid in ['q1', 'q2', 'q3']:
        for i in range(10):
            data.append({
                'qid': qid,
                'query': f'query {qid}',
                'docno': f'{qid}_d{i}',
                'text': f'document {i} for {qid}',
                'score': 10 - i
            })
    return pd.DataFrame(data)


# Tests for GenerativeRanker initialization

class TestGenerativeRankerInit:
    """Test GenerativeRanker initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        assert ranker.model is backend
        assert ranker.algorithm == Algorithm.SINGLE_WINDOW
        assert ranker.window_size == 20  # default

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Custom prompt",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=15,
            stride=7,
            buffer=25,
            cutoff=8,
            k=12,
            max_iters=50
        )

        assert ranker.window_size == 15
        assert ranker.stride == 7
        assert ranker.buffer == 25
        assert ranker.cutoff == 8
        assert ranker.k == 12
        assert ranker.max_iters == 50

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        backend = MessageBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="User prompt",
            system_prompt="System instructions",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        assert ranker.system_prompt == "System instructions"

    def test_init_with_callable_prompt(self):
        """Test initialization with callable prompt."""
        backend = DeterministicBackend()

        def custom_prompt(query, passages, num):
            return f"Q: {query}\nDocs: {num}"

        ranker = GenerativeRanker(
            model=backend,
            prompt=custom_prompt,
            algorithm=Algorithm.SINGLE_WINDOW
        )

        assert callable(ranker.prompt)


# Tests for prompt construction

class TestPromptConstruction:
    """Test prompt construction methods."""

    def test_string_prompt_basic(self):
        """Test basic string prompt construction."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        # Create mock doc rows
        class DocRow:
            def __init__(self, text):
                self.text = text

        docs = [(0, DocRow("doc0")), (1, DocRow("doc1")), (2, DocRow("doc2"))]

        prompt = ranker.make_prompt_from(
            docs=docs,
            query="test",
            num=3,
            passages=["doc0", "doc1", "doc2"]
        )

        assert "Query: test" in prompt
        assert "[1] doc0" in prompt
        assert "[2] doc1" in prompt
        assert "[3] doc2" in prompt

    def test_callable_prompt(self):
        """Test callable prompt."""
        backend = DeterministicBackend()

        def custom_prompt(query, passages, num):
            return f"Q={query} N={num} P={len(passages)}"

        ranker = GenerativeRanker(
            model=backend,
            prompt=custom_prompt,
            algorithm=Algorithm.SINGLE_WINDOW
        )

        class DocRow:
            def __init__(self, text):
                self.text = text

        docs = [(0, DocRow("doc0")), (1, DocRow("doc1"))]

        prompt = ranker.make_prompt_from(
            docs=docs,
            query="test",
            num=2,
            passages=["doc0", "doc1"]
        )

        assert "Q=test" in prompt
        assert "N=2" in prompt
        assert "P=2" in prompt

    def test_message_prompt(self):
        """Test message-based prompt construction."""
        backend = MessageBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}",
            system_prompt="System message",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        class DocRow:
            def __init__(self, text):
                self.text = text

        docs = [(0, DocRow("doc0"))]

        prompt = ranker.make_prompt_from(
            docs=docs,
            query="test",
            num=1,
            passages=["doc0"]
        )

        # For message-based backends, should return messages list
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0] == {'role': 'system', 'content': 'System message'}
        assert prompt[1]['role'] == 'user'
        assert "Query: test" in prompt[1]['content']


# Tests for output parsing

class TestOutputParsing:
    """Test output parsing logic."""

    def test_parse_simple_output(self):
        """Test parsing simple sequential output."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        order = ranker.parse_output("1 2 3 4 5", length=5)
        assert order == [0, 1, 2, 3, 4]  # 0-indexed

    def test_parse_reverse_output(self):
        """Test parsing reverse order."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        order = ranker.parse_output("5 4 3 2 1", length=5)
        assert order == [4, 3, 2, 1, 0]

    def test_parse_with_duplicates(self):
        """Test parsing with duplicate indices."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        # Duplicates should be removed (first occurrence kept)
        order = ranker.parse_output("1 2 2 3 1 4", length=5)
        assert len(order) == 5
        assert len(set(order)) == 5  # All unique after backfill

    def test_parse_with_missing(self):
        """Test parsing with missing indices."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        # Missing indices should be backfilled
        order = ranker.parse_output("1 3 5", length=5)
        assert len(order) == 5
        assert 0 in order  # 1-indexed becomes 0
        assert 2 in order  # 3-indexed becomes 2
        assert 4 in order  # 5-indexed becomes 4

    def test_parse_with_noise(self):
        """Test parsing with noisy output."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        # Should extract only numbers
        order = ranker.parse_output("The ranking is: [1], [2], [3]", length=3)
        assert order == [0, 1, 2]

    def test_parse_out_of_range(self):
        """Test parsing with out-of-range indices."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        # Out of range indices should be filtered out
        order = ranker.parse_output("1 2 10 20 3", length=5)
        assert len(order) == 5
        assert 9 not in order  # 10-indexed
        assert 19 not in order  # 20-indexed


# Tests for transform method

class TestTransform:
    """Test transform method."""

    def test_basic_transform(self, sample_data):
        """Test basic transformation."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        result = ranker.transform(sample_data)

        # Check output structure
        assert len(result) == 10
        assert all(col in result.columns for col in ['qid', 'query', 'docno', 'text', 'rank', 'score'])
        assert result['qid'].iloc[0] == 'q1'
        assert result['query'].iloc[0] == 'test query'

    def test_transform_multi_query(self, multi_query_data):
        """Test transformation with multiple queries."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        result = ranker.transform(multi_query_data)

        # Should have results for all queries
        assert len(result) == 30  # 3 queries * 10 docs
        assert set(result['qid'].unique()) == {'q1', 'q2', 'q3'}

        # Each query should have 10 results
        for qid in ['q1', 'q2', 'q3']:
            query_results = result[result['qid'] == qid]
            assert len(query_results) == 10

    def test_transform_empty_input(self):
        """Test transformation with empty input."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        empty_df = pd.DataFrame(columns=['qid', 'query', 'docno', 'text', 'score'])
        result = ranker.transform(empty_df)

        assert len(result) == 0
        assert all(col in result.columns for col in ['qid', 'query', 'docno', 'text', 'rank', 'score'])

    def test_transform_sliding_window(self, sample_data):
        """Test transformation with sliding window algorithm."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SLIDING_WINDOW,
            window_size=5,
            stride=3
        )

        result = ranker.transform(sample_data)

        assert len(result) == 10
        assert all(col in result.columns for col in ['qid', 'query', 'docno', 'text', 'rank', 'score'])

    def test_transform_ranks_and_scores(self, sample_data):
        """Test that ranks and scores are properly assigned."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW,
            window_size=20
        )

        result = ranker.transform(sample_data)

        # Ranks should be sequential starting from 0
        assert list(result['rank']) == list(range(10))

        # Scores should be decreasing (higher rank = lower score)
        scores = result['score'].tolist()
        assert scores == sorted(scores, reverse=True)


# Tests for batching

class TestBatching:
    """Test batched window ranking."""

    def test_rank_windows_batch(self):
        """Test batch ranking of multiple windows."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        windows_kwargs = [
            {
                'query': 'test',
                'doc_text': ['doc0', 'doc1', 'doc2'],
                'window_len': 3
            },
            {
                'query': 'test',
                'doc_text': ['doc3', 'doc4'],
                'window_len': 2
            }
        ]

        orders = ranker._rank_windows_batch(windows_kwargs)

        assert len(orders) == 2
        assert len(orders[0]) == 3
        assert len(orders[1]) == 2

        # Should have made single batched call
        assert len(backend.generate_calls) == 1
        assert backend.generate_calls[0] == 2  # 2 prompts in batch

    def test_batch_empty_input(self):
        """Test batching with empty input."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="test",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        orders = ranker._rank_windows_batch([])
        assert orders == []


# Tests for callable interface

class TestCallable:
    """Test that ranker is callable for algorithms."""

    def test_ranker_is_callable(self):
        """Test that ranker can be called like a function."""
        backend = DeterministicBackend()
        ranker = GenerativeRanker(
            model=backend,
            prompt="Query: {{ query }}\n{% for p in passages %}[{{ loop.index }}] {{ p }}\n{% endfor %}",
            algorithm=Algorithm.SINGLE_WINDOW
        )

        result = ranker(
            query='test',
            doc_text=['doc0', 'doc1', 'doc2'],
            window_len=3
        )

        assert isinstance(result, list)
        assert len(result) == 3
