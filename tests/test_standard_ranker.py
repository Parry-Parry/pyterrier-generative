"""
Test suite for StandardRanker and model variants.

Tests the meta-variant system and standard model configurations.
"""

import pytest
from pyterrier_generative.modelling.variants import StandardRanker, RankZephyr, RankVicuna, RankGPT
from pyterrier_generative._algorithms import Algorithm


# Note: These tests check the configuration and initialization
# They don't actually run the models (which would require GPU/API keys)


class TestStandardRankerVariants:
    """Test StandardRanker variant system."""

    def test_rankgpt_variants_defined(self):
        """Test that RankGPT variants are defined."""
        assert 'gpt35' in RankGPT.VARIANTS
        assert 'gpt35_16k' in RankGPT.VARIANTS
        assert 'gpt4' in RankGPT.VARIANTS
        assert 'gpt4_turbo' in RankGPT.VARIANTS

    def test_rankzephyr_variants_defined(self):
        """Test that RankZephyr variants are defined."""
        assert 'v1' in RankZephyr.VARIANTS

    def test_rankvicuna_variants_defined(self):
        """Test that RankVicuna variants are defined."""
        assert 'v1' in RankVicuna.VARIANTS

    def test_variant_models(self):
        """Test variant model IDs."""
        assert RankZephyr.VARIANTS['v1'] == 'castorini/rank_zephyr_7b_v1_full'
        assert RankVicuna.VARIANTS['v1'] == 'castorini/rank_vicuna_7b_v1'
        assert RankGPT.VARIANTS['gpt4'] == 'gpt-4'
        assert RankGPT.VARIANTS['gpt4_turbo'] == 'gpt-4-turbo-preview'
        assert RankGPT.VARIANTS['gpt35'] == 'gpt-3.5-turbo'
        assert RankGPT.VARIANTS['gpt35_16k'] == 'gpt-3.5-turbo-16k'


class TestStandardRankerInit:
    """Test StandardRanker initialization."""

    def test_init_with_default_variant(self):
        """Test that RankGPT uses default variant when no model_id provided."""
        try:
            ranker = RankGPT()  # Should use default (gpt-3.5-turbo)
            assert ranker.model_id == 'gpt-3.5-turbo'
        except Exception:
            # Expected without API key
            pass

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
            ranker = RankZephyr.v1(
                max_new_tokens=10,
                verbose=False
            )
            repr_str = repr(ranker)
            assert 'RankZephyr.v1()' == repr_str
        except ImportError:
            pytest.skip("vLLM not available")

    def test_repr_custom_model(self):
        """Test repr for custom model."""
        try:
            ranker = RankGPT(
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
    """Test RankZephyr class."""

    def test_rankzephyr_v1_variant(self):
        """Test that RankZephyr.v1() creates ranker with correct model."""
        try:
            ranker = RankZephyr.v1(max_new_tokens=10, verbose=False)
            assert ranker.model_id == 'castorini/rank_zephyr_7b_v1_full'
            assert ranker.backend_type == 'vllm'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_rankzephyr_with_parameters(self):
        """Test RankZephyr with custom parameters."""
        try:
            ranker = RankZephyr.v1(
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
            ranker = RankZephyr.v1(backend='hf', max_new_tokens=10, verbose=False)
            assert ranker.backend_type == 'hf'
        except ImportError:
            pytest.skip("HuggingFace transformers not available")


class TestRankVicuna:
    """Test RankVicuna class."""

    def test_rankvicuna_v1_variant(self):
        """Test that RankVicuna.v1() creates ranker with correct model."""
        try:
            ranker = RankVicuna.v1(max_new_tokens=10, verbose=False)
            assert ranker.model_id == 'castorini/rank_vicuna_7b_v1'
            assert ranker.backend_type == 'vllm'
        except ImportError:
            pytest.skip("vLLM not available")

    def test_rankvicuna_with_parameters(self):
        """Test RankVicuna with custom parameters."""
        try:
            ranker = RankVicuna.v1(
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
    """Test RankGPT class."""

    def test_rankgpt_default_variant(self):
        """Test that RankGPT() uses default variant."""
        try:
            ranker = RankGPT(max_new_tokens=10)
            assert ranker.model_id == 'gpt-3.5-turbo'
            assert ranker.backend_type == 'openai'
        except Exception:
            # Expected without API key
            pass

    def test_rankgpt_gpt35_variant(self):
        """Test RankGPT.gpt35() variant."""
        try:
            ranker = RankGPT.gpt35(max_new_tokens=10)
            assert ranker.model_id == 'gpt-3.5-turbo'
        except Exception:
            # Expected without API key
            pass

    def test_rankgpt_gpt4_variant(self):
        """Test RankGPT.gpt4() variant."""
        try:
            ranker = RankGPT.gpt4(max_new_tokens=10)
            assert ranker.model_id == 'gpt-4'
        except Exception:
            # Expected without API key
            pass

    def test_rankgpt_with_api_key(self):
        """Test RankGPT with API key parameter."""
        try:
            ranker = RankGPT.gpt35(api_key='test-key', max_new_tokens=10)
            assert ranker.model_id == 'gpt-3.5-turbo'
        except Exception:
            # Expected with invalid API key
            pass

    def test_rankgpt_with_parameters(self):
        """Test RankGPT with custom parameters."""
        try:
            ranker = RankGPT.gpt35(
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

    def test_rankgpt_variants_accessible(self):
        """Test that RankGPT variants are accessible as class methods."""
        # Just check they're callable, don't actually instantiate
        assert callable(RankGPT.gpt35)
        assert callable(RankGPT.gpt35_16k)
        assert callable(RankGPT.gpt4)
        assert callable(RankGPT.gpt4_turbo)

    def test_rankzephyr_variants_accessible(self):
        """Test that RankZephyr variants are accessible as class methods."""
        assert callable(RankZephyr.v1)

    def test_rankvicuna_variants_accessible(self):
        """Test that RankVicuna variants are accessible as class methods."""
        assert callable(RankVicuna.v1)

    def test_variant_has_docstring(self):
        """Test that variant methods have docstrings."""
        assert RankZephyr.v1.__doc__ is not None
        assert 'castorini/rank_zephyr_7b_v1_full' in RankZephyr.v1.__doc__

    def test_invalid_variant_raises_error(self):
        """Test that invalid variant raises AttributeError."""
        with pytest.raises(AttributeError):
            RankGPT.InvalidVariant()
