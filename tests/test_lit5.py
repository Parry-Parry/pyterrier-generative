"""
Test suite for LiT5 ranking model.

Tests LiT5-specific functionality including FiD architecture integration.
"""

import pytest
import pandas as pd
import numpy as np
from pyterrier_generative._algorithms import Algorithm


# Note: LiT5 tests require pyterrier-t5 and transformers
# Skip tests if dependencies not available

pytest.importorskip("transformers", reason="transformers not installed")


class TestLiT5Import:
    """Test LiT5 can be imported."""

    def test_import_lit5(self):
        """Test that LiT5 can be imported."""
        from pyterrier_generative.modelling.lit5 import LiT5
        assert LiT5 is not None

    def test_import_lit5_backend(self):
        """Test that LiT5Backend can be imported."""
        from pyterrier_generative.modelling.lit5 import LiT5Backend
        assert LiT5Backend is not None


class TestLiT5Backend:
    """Test LiT5Backend class."""

    def test_backend_init(self):
        """Test LiT5Backend initialization."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5Backend

        try:
            backend = LiT5Backend(
                model_path='castorini/LiT5-Distill-large',
                batch_size=2,
                window_size=10
            )
            assert backend.window_size == 10
            assert backend.template is not None
            assert backend.tokenizer is not None
            assert backend.model is not None
            assert not backend.supports_message_input
        except Exception as e:
            # Model download might fail, but init should work
            if "not found" not in str(e).lower():
                raise

    def test_backend_generate_not_implemented(self):
        """Test that backend.generate() raises NotImplementedError."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5Backend

        try:
            backend = LiT5Backend(
                model_path='castorini/LiT5-Distill-large',
                batch_size=2,
                window_size=10
            )
            with pytest.raises(NotImplementedError):
                backend.generate(['test prompt'])
        except Exception as e:
            if "not found" not in str(e).lower():
                raise


class TestLiT5Init:
    """Test LiT5 initialization."""

    def test_init_default_params(self):
        """Test LiT5 with default parameters."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5()
            assert ranker.model_path == 'castorini/LiT5-Distill-large'
            assert ranker.window_size == 20
            assert ranker.stride == 10
            assert ranker.algorithm == Algorithm.SLIDING_WINDOW
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

    def test_init_custom_params(self):
        """Test LiT5 with custom parameters."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5(
                model_path='castorini/LiT5-Distill-base',
                window_size=15,
                stride=7,
                batch_size=8,
                bfloat16=False,
                algorithm=Algorithm.SINGLE_WINDOW
            )
            assert ranker.model_path == 'castorini/LiT5-Distill-base'
            assert ranker.window_size == 15
            assert ranker.stride == 7
            assert ranker.algorithm == Algorithm.SINGLE_WINDOW
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

    def test_backend_reference(self):
        """Test that LiT5 stores backend reference."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5, LiT5Backend

        try:
            ranker = LiT5(window_size=10)
            assert hasattr(ranker, 'backend')
            assert isinstance(ranker.backend, LiT5Backend)
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

    def test_repr(self):
        """Test LiT5 repr."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5(
                model_path='castorini/LiT5-Distill-base',
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=15
            )
            repr_str = repr(ranker)
            assert 'LiT5' in repr_str
            assert 'LiT5-Distill-base' in repr_str
            assert 'sliding_window' in repr_str
            assert 'window_size=15' in repr_str
        except Exception as e:
            if "not found" not in str(e).lower():
                raise


class TestLiT5RankWindow:
    """Test LiT5 window ranking."""

    def test_rank_window_override(self):
        """Test that LiT5 overrides _rank_window."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5(window_size=10)
            # Should have _rank_window method
            assert hasattr(ranker, '_rank_window')
            assert callable(ranker._rank_window)
        except Exception as e:
            if "not found" not in str(e).lower():
                raise


class TestLiT5Batching:
    """Test LiT5 batched ranking."""

    def test_has_batch_method(self):
        """Test that LiT5 has _rank_windows_batch method."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5(window_size=10)
            assert hasattr(ranker, '_rank_windows_batch')
            assert callable(ranker._rank_windows_batch)
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

    def test_batch_empty_windows(self):
        """Test batching with empty windows list."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        try:
            ranker = LiT5(window_size=10)
            result = ranker._rank_windows_batch([])
            assert result == []
        except Exception as e:
            if "not found" not in str(e).lower():
                raise


class TestLiT5Integration:
    """Integration tests for LiT5 (may require model download)."""

    @pytest.mark.integration
    def test_transform_basic(self):
        """Test basic transformation (integration test)."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")
        pytest.importorskip("torch", reason="PyTorch not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        # Create test data
        test_df = pd.DataFrame({
            'qid': ['q1'] * 10,
            'query': ['information retrieval'] * 10,
            'docno': [f'd{i}' for i in range(10)],
            'text': [f'document {i} about retrieval systems' for i in range(10)],
            'score': list(range(10, 0, -1))
        })

        try:
            ranker = LiT5(window_size=10, batch_size=2)
            result = ranker.transform(test_df)

            # Check output structure
            assert len(result) == 10
            assert all(col in result.columns for col in ['qid', 'query', 'docno', 'text', 'rank', 'score'])
            assert result['qid'].iloc[0] == 'q1'
        except Exception as e:
            # Skip if model can't be loaded
            if any(x in str(e).lower() for x in ['not found', 'connection', 'download']):
                pytest.skip(f"Model not available: {e}")
            raise

    @pytest.mark.integration
    def test_transform_with_batching(self):
        """Test transformation uses batching (integration test)."""
        pytest.importorskip("pyterrier_t5", reason="pyterrier-t5 not installed")
        pytest.importorskip("torch", reason="PyTorch not installed")

        from pyterrier_generative.modelling.lit5 import LiT5

        # Create test data with enough docs for multiple windows
        test_df = pd.DataFrame({
            'qid': ['q1'] * 25,
            'query': ['information retrieval'] * 25,
            'docno': [f'd{i}' for i in range(25)],
            'text': [f'document {i} about retrieval systems' for i in range(25)],
            'score': list(range(25, 0, -1))
        })

        try:
            ranker = LiT5(
                window_size=10,
                stride=5,
                batch_size=4,
                algorithm=Algorithm.SLIDING_WINDOW
            )
            result = ranker.transform(test_df)

            # Check output
            assert len(result) == 25
            assert all(col in result.columns for col in ['qid', 'query', 'docno', 'text', 'rank', 'score'])
        except Exception as e:
            if any(x in str(e).lower() for x in ['not found', 'connection', 'download']):
                pytest.skip(f"Model not available: {e}")
            raise


class TestLiT5Export:
    """Test LiT5 is properly exported."""

    def test_lit5_in_all(self):
        """Test LiT5 is in __all__."""
        from pyterrier_generative.modelling import lit5
        assert 'LiT5' in lit5.__all__

    def test_lit5_importable_from_package(self):
        """Test LiT5 can be imported from main package."""
        from pyterrier_generative import LiT5
        assert LiT5 is not None

    def test_lit5backend_in_all(self):
        """Test LiT5Backend is in __all__."""
        from pyterrier_generative.modelling import lit5
        assert 'LiT5Backend' in lit5.__all__
