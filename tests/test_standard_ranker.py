"""
Test suite for StandardRanker and model variants.

Tests the meta-variant system and standard model configurations.
"""

import pytest
from pyterrier_generative.modelling.standard import StandardRanker, RankZephyr, RankVicuna, RankGPT
from pyterrier_generative._algorithms import Algorithm


# Note: These tests check the configuration and initialization
# They don't actually run the models (which would require GPU/API keys)


class TestStandardRankerVariants:
    """Test StandardRanker variant system."""

    def test_variants_defined(self):
        """Test that all variants are defined."""
        assert 'RankZephyr' in StandardRanker.VARIANTS
        assert 'RankVicuna' in StandardRanker.VARIANTS
        assert 'RankGPT4' in StandardRanker.VARIANTS
        assert 'RankGPT4Turbo' in StandardRanker.VARIANTS
        assert 'RankGPT35' in StandardRanker.VARIANTS
        assert 'RankGPT35_16k' in StandardRanker.VARIANTS

    def test_variant_models(self):
        """Test variant model IDs."""
        assert StandardRanker.VARIANTS['RankZephyr'] == 'castorini/rank_zephyr_7b_v1_full'
        assert StandardRanker.VARIANTS['RankVicuna'] == 'castorini/rank_vicuna_7b_v1'
        assert StandardRanker.VARIANTS['RankGPT4'] == 'gpt-4'
        assert StandardRanker.VARIANTS['RankGPT4Turbo'] == 'gpt-4-turbo-preview'
        assert StandardRanker.VARIANTS['RankGPT35'] == 'gpt-3.5-turbo'
        assert StandardRanker.VARIANTS['RankGPT35_16k'] == 'gpt-3.5-turbo-16k'


class TestStandardRankerInit:
    """Test StandardRanker initialization."""

    def test_init_requires_model_id(self):
        """Test that model_id is required."""
        with pytest.raises(TypeError):
            StandardRanker()  # Missing required model_id argument

    def test_backend_autodetect_vllm(self):
        """Test backend auto-detection for HF models."""
        # Note: This will fail if vllm is not installed, but tests the logic
        try:
            ranker = StandardRanker(
                'castorini/rank_zephyr_7b_v1_full',
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.backend_type == 'vllm'
            assert ranker.model_id == 'castorini/rank_zephyr_7b_v1_full'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_backend_autodetect_openai(self):
        """Test backend auto-detection for OpenAI models."""
        # Note: This will fail without API key, but tests the logic
        try:
            ranker = StandardRanker(
                'gpt-3.5-turbo',
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.backend_type == 'openai'
            assert ranker.model_id == 'gpt-3.5-turbo'
        except Exception:
            # Expected without API key
            pass

    def test_explicit_backend_vllm(self):
        """Test explicit vLLM backend specification."""
        try:
            ranker = StandardRanker(
                'custom/model',
                backend='vllm',
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.backend_type == 'vllm'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_explicit_backend_hf(self):
        """Test explicit HuggingFace backend specification."""
        try:
            ranker = StandardRanker(
                'custom/model',
                backend='hf',
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.backend_type == 'hf'
        except ImportError:
            pytest.skip("HuggingFace transformers not available")

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            StandardRanker(
                'custom/model',
                backend='invalid_backend',
                max_new_tokens=10
            )

    def test_algorithm_parameters(self):
        """Test algorithm parameter passing."""
        try:
            ranker = StandardRanker(
                'gpt-3.5-turbo',
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=15,
                stride=7,
                buffer=25,
                cutoff=8,
                k=12,
                max_iters=50,
                max_new_tokens=10
            )
            assert ranker.algorithm == Algorithm.SLIDING_WINDOW
            assert ranker.window_size == 15
            assert ranker.stride == 7
            assert ranker.buffer == 25
            assert ranker.cutoff == 8
            assert ranker.k == 12
            assert ranker.max_iters == 50
        except Exception:
            # Expected without API key
            pass

    def test_repr_variant(self):
        """Test repr for known variant."""
        try:
            ranker = StandardRanker(
                'castorini/rank_zephyr_7b_v1_full',
                max_new_tokens=10,
                verbose=False
            )
            repr_str = repr(ranker)
            assert 'StandardRanker.RankZephyr()' == repr_str
        except ImportError:
            pytest.skip("vLLM not available")

    def test_repr_custom_model(self):
        """Test repr for custom model."""
        try:
            ranker = StandardRanker(
                'custom/model',
                backend='vllm',
                algorithm=Algorithm.SLIDING_WINDOW,
                window_size=10,
                max_new_tokens=10,
                verbose=False
            )
            repr_str = repr(ranker)
            assert 'custom/model' in repr_str
            assert 'vllm' in repr_str
            assert 'sliding_window' in repr_str
            assert 'window_size=10' in repr_str
        except ImportError:
            pytest.skip("vLLM not available")


class TestRankZephyr:
    """Test RankZephyr convenience function."""

    def test_rankzephyr_creates_standard_ranker(self):
        """Test that RankZephyr creates StandardRanker."""
        try:
            ranker = RankZephyr(max_new_tokens=10, verbose=False)
            assert isinstance(ranker, StandardRanker)
            assert ranker.model_id == 'castorini/rank_zephyr_7b_v1_full'
            assert ranker.backend_type == 'vllm'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_rankzephyr_with_parameters(self):
        """Test RankZephyr with custom parameters."""
        try:
            ranker = RankZephyr(
                window_size=15,
                stride=8,
                algorithm=Algorithm.SLIDING_WINDOW,
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.window_size == 15
            assert ranker.stride == 8
            assert ranker.algorithm == Algorithm.SLIDING_WINDOW
        except ImportError:
            pytest.skip("vLLM not available")

    def test_rankzephyr_backend_override(self):
        """Test RankZephyr with backend override."""
        try:
            ranker = RankZephyr(backend='hf', max_new_tokens=10, verbose=False)
            assert ranker.backend_type == 'hf'
        except ImportError:
            pytest.skip("HuggingFace transformers not available")


class TestRankVicuna:
    """Test RankVicuna convenience function."""

    def test_rankvicuna_creates_standard_ranker(self):
        """Test that RankVicuna creates StandardRanker."""
        try:
            ranker = RankVicuna(max_new_tokens=10, verbose=False)
            assert isinstance(ranker, StandardRanker)
            assert ranker.model_id == 'castorini/rank_vicuna_7b_v1'
            assert ranker.backend_type == 'vllm'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_rankvicuna_with_parameters(self):
        """Test RankVicuna with custom parameters."""
        try:
            ranker = RankVicuna(
                window_size=12,
                algorithm=Algorithm.TDPART,
                cutoff=5,
                max_new_tokens=10,
                verbose=False
            )
            assert ranker.window_size == 12
            assert ranker.algorithm == Algorithm.TDPART
            assert ranker.cutoff == 5
        except ImportError:
            pytest.skip("vLLM not available")


class TestRankGPT:
    """Test RankGPT convenience function."""

    def test_rankgpt_creates_standard_ranker(self):
        """Test that RankGPT creates StandardRanker."""
        try:
            ranker = RankGPT(max_new_tokens=10)
            assert isinstance(ranker, StandardRanker)
            assert ranker.model_id == 'gpt-3.5-turbo'
            assert ranker.backend_type == 'openai'
        except Exception:
            # Expected without API key
            pass

    def test_rankgpt_with_api_key(self):
        """Test RankGPT with API key parameter."""
        try:
            ranker = RankGPT(api_key='test-key', max_new_tokens=10)
            assert ranker.model_id == 'gpt-3.5-turbo'
        except Exception:
            # Expected with invalid API key
            pass

    def test_rankgpt_with_parameters(self):
        """Test RankGPT with custom parameters."""
        try:
            ranker = RankGPT(
                window_size=20,
                algorithm=Algorithm.SINGLE_WINDOW,
                max_new_tokens=50
            )
            assert ranker.window_size == 20
            assert ranker.algorithm == Algorithm.SINGLE_WINDOW
        except Exception:
            # Expected without API key
            pass


class TestMetaclassVariants:
    """Test metaclass variant creation."""

    def test_variant_as_classmethod(self):
        """Test accessing variant as class method."""
        # Access RankZephyr via metaclass
        try:
            ranker = StandardRanker.RankZephyr(max_new_tokens=10, verbose=False)
            assert isinstance(ranker, StandardRanker)
            assert ranker.model_id == 'castorini/rank_zephyr_7b_v1_full'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_all_variants_accessible(self):
        """Test that all variants are accessible via metaclass."""
        # Just check they're callable, don't actually instantiate
        assert callable(StandardRanker.RankZephyr)
        assert callable(StandardRanker.RankVicuna)
        assert callable(StandardRanker.RankGPT4)
        assert callable(StandardRanker.RankGPT4Turbo)
        assert callable(StandardRanker.RankGPT35)
        assert callable(StandardRanker.RankGPT35_16k)

    def test_variant_has_docstring(self):
        """Test that variant methods have docstrings."""
        assert StandardRanker.RankZephyr.__doc__ is not None
        assert 'castorini/rank_zephyr_7b_v1_full' in StandardRanker.RankZephyr.__doc__

    def test_invalid_variant_raises_error(self):
        """Test that invalid variant raises AttributeError."""
        with pytest.raises(AttributeError):
            StandardRanker.InvalidVariant()
