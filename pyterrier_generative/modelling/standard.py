"""Pre-configured standard ranking models."""

from typing import Optional, Union
import torch

from pyterrier_generative.modelling.base import GenerativeRanker
from pyterrier_generative._algorithms import Algorithm
from pyterrier_generative.prompts import RANKPROMPT
from pyterrier_generative.modelling.util import Variants


class StandardRanker(GenerativeRanker, metaclass=Variants):
    """
    Standard pre-configured rankers with multiple model variants.

    This class provides easy access to common ranking models through
    class attributes that act as factory methods.

    Available Models:

    - **RankZephyr**: ``castorini/rank_zephyr_7b_v1_full`` (vLLM)
    - **RankVicuna**: ``castorini/rank_vicuna_7b_v1`` (vLLM)
    - **RankGPT4**: ``gpt-4`` (OpenAI)
    - **RankGPT4Turbo**: ``gpt-4-turbo-preview`` (OpenAI)
    - **RankGPT35**: ``gpt-3.5-turbo`` (OpenAI)
    - **RankGPT35_16k**: ``gpt-3.5-turbo-16k`` (OpenAI)

    Example::

        from pyterrier_generative import StandardRanker
        import pyterrier as pt

        # Use RankZephyr
        ranker = StandardRanker.RankZephyr()

        # Use RankGPT with GPT-4
        ranker = StandardRanker.RankGPT4(window_size=10)

        # Use RankVicuna with custom parameters
        ranker = StandardRanker.RankVicuna(stride=5, backend='hf')

        # In a pipeline
        pipeline = bm25 % 20 >> StandardRanker.RankZephyr()
        results = pipeline.search("What is information retrieval?")

    .. automethod:: RankZephyr()
    .. automethod:: RankVicuna()
    .. automethod:: RankGPT4()
    .. automethod:: RankGPT4Turbo()
    .. automethod:: RankGPT35()
    .. automethod:: RankGPT35_16k()
    """

    VARIANTS = {
        'RankZephyr': 'castorini/rank_zephyr_7b_v1_full',
        'RankVicuna': 'castorini/rank_vicuna_7b_v1',
        'RankGPT4': 'gpt-4',
        'RankGPT4Turbo': 'gpt-4-turbo-preview',
        'RankGPT35': 'gpt-3.5-turbo',
        'RankGPT35_16k': 'gpt-3.5-turbo-16k',
    }

    def __init__(
        self,
        model_id: str,
        *,
        prompt: Union[str, callable] = RANKPROMPT,
        system_prompt: str = "",
        algorithm: Algorithm = Algorithm.SLIDING_WINDOW,
        window_size: int = 20,
        stride: int = 10,
        buffer: int = 20,
        cutoff: int = 10,
        k: int = 10,
        max_iters: int = 100,
        max_new_tokens: int = 100,
        backend: Optional[str] = None,
        model_args: Optional[dict] = None,
        generation_args: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize StandardRanker with the specified model."""

        # Auto-detect backend based on model_id if not specified
        if backend is None:
            if model_id.startswith('gpt-'):
                backend = 'openai'
            else:
                backend = 'vllm'

        # Handle API key for OpenAI
        if api_key and backend == 'openai':
            generation_args = generation_args or {}
            generation_args['api_key'] = api_key

        # Select and initialize backend
        if backend == 'vllm':
            from pyterrier_rag.backend.vllm import VLLMBackend
            backend_instance = VLLMBackend(
                model_id=model_id,
                model_args=model_args or {},
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        elif backend == 'hf':
            from pyterrier_rag.backend.hf import HuggingFaceBackend
            backend_instance = HuggingFaceBackend(
                model_id=model_id,
                model_args=model_args or {},
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                device=device,
                verbose=verbose,
            )
        elif backend == 'openai':
            from pyterrier_rag.backend.openai import OpenAIBackend
            backend_instance = OpenAIBackend(
                model_id=model_id,
                generation_args=generation_args,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'vllm', 'hf', or 'openai'.")

        # Initialize parent GenerativeRanker
        super().__init__(
            model=backend_instance,
            prompt=prompt,
            system_prompt=system_prompt,
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
            buffer=buffer,
            cutoff=cutoff,
            k=k,
            max_iters=max_iters,
        )

        self.model_id = model_id
        self.backend_type = backend

    def __repr__(self):
        # Check if this is a known variant
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_id in inv_variants:
            return f'StandardRanker.{inv_variants[self.model_id]}()'

        return (
            f"StandardRanker("
            f"model_id={self.model_id!r}, "
            f"backend={self.backend_type!r}, "
            f"algorithm={self.algorithm.value!r}, "
            f"window_size={self.window_size})"
        )


# Create convenient aliases at module level
RankZephyr = lambda **kwargs: StandardRanker(StandardRanker.VARIANTS['RankZephyr'], **kwargs)
RankVicuna = lambda **kwargs: StandardRanker(StandardRanker.VARIANTS['RankVicuna'], **kwargs)
RankGPT = lambda **kwargs: StandardRanker(StandardRanker.VARIANTS['RankGPT35'], **kwargs)

# Set docstrings for the aliases
RankZephyr.__doc__ = """Alias for StandardRanker.RankZephyr(). Model: ``castorini/rank_zephyr_7b_v1_full``"""
RankVicuna.__doc__ = """Alias for StandardRanker.RankVicuna(). Model: ``castorini/rank_vicuna_7b_v1``"""
RankGPT.__doc__ = """Alias for StandardRanker.RankGPT35(). Model: ``gpt-3.5-turbo``"""


__all__ = ['StandardRanker', 'RankZephyr', 'RankVicuna', 'RankGPT']
